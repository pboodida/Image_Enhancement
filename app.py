import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

st.set_page_config(
    page_title="Low Light Image Enhancement",
    page_icon="💡",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4a4a4a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666666;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Low Light Image Enhancement</h1>", unsafe_allow_html=True)
st.markdown("Transform your dark and low-quality images into bright, clear photos with our AI enhancement model.")

# --- Custom Losses and Metrics ---
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim = ssim_loss(y_true, y_pred)
    return 0.4 * mse + 0.6 * ssim

def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

# --- Image Preprocessing ---
def preprocess_image(image, target_size=(384, 384)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    original_size = image.size
    resized_image = image.resize(target_size)
    img_array = np.array(resized_image).astype(np.float32) / 255.0
    return img_array, original_size

# --- Attention Modules ---
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    shared_layer_one = tf.keras.layers.Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
    shared_layer_two = tf.keras.layers.Dense(channel, kernel_initializer='he_normal', use_bias=True)

    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = tf.keras.layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = tf.keras.layers.GlobalMaxPooling2D()(input_feature)
    max_pool = tf.keras.layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = tf.keras.layers.Add()([avg_pool, max_pool])
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)

    return tf.keras.layers.Multiply()([input_feature, cbam_feature])

def spatial_attention(input_feature):
    avg_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(input_feature)
    max_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(input_feature)
    concat = tf.keras.layers.Concatenate()([avg_pool, max_pool])
    cbam_feature = tf.keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    return tf.keras.layers.Multiply()([input_feature, cbam_feature])

def attention_gate(x, gating, inter_channels):
    theta_x = tf.keras.layers.Conv2D(inter_channels, 1, padding='same')(x)
    phi_g = tf.keras.layers.Conv2D(inter_channels, 1, padding='same')(gating)

    if theta_x.shape[1] != phi_g.shape[1] or theta_x.shape[2] != phi_g.shape[2]:
        phi_g = tf.keras.layers.UpSampling2D(
            size=(theta_x.shape[1] // phi_g.shape[1], theta_x.shape[2] // phi_g.shape[2])
        )(phi_g)

    add = tf.keras.layers.Add()([theta_x, phi_g])
    relu = tf.keras.layers.Activation('relu')(add)
    psi = tf.keras.layers.Conv2D(1, kernel_size=1, strides=1, padding='same')(relu)
    sigmoid = tf.keras.layers.Activation('sigmoid')(psi)
    return tf.keras.layers.Multiply()([x, sigmoid])

def residual_block(input_tensor, filters, kernel_size=3):
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = channel_attention(x)
    x = spatial_attention(x)

    if input_tensor.shape[-1] != filters:
        input_tensor = tf.keras.layers.Conv2D(filters, 1, padding='same')(input_tensor)

    x = tf.keras.layers.Add()([x, input_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x

# --- Model Builder ---
def build_enhanced_unet(input_shape=(None, None, 3), num_filters=48):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    conv1 = x

    pool1 = tf.keras.layers.MaxPooling2D(2)(conv1)
    conv2 = residual_block(pool1, num_filters*2)

    pool2 = tf.keras.layers.MaxPooling2D(2)(conv2)
    x = residual_block(pool2, num_filters*4)
    conv3 = residual_block(x, num_filters*4)

    pool3 = tf.keras.layers.MaxPooling2D(2)(conv3)
    x = residual_block(pool3, num_filters*8)
    conv4 = residual_block(x, num_filters*8)

    pool4 = tf.keras.layers.MaxPooling2D(2)(conv4)
    x = residual_block(pool4, num_filters*16)
    bottle = residual_block(x, num_filters*16)

    up4 = tf.keras.layers.UpSampling2D(2)(bottle)
    attn4 = attention_gate(conv4, up4, num_filters*4)
    concat4 = tf.keras.layers.Concatenate()([up4, attn4])
    deconv4 = residual_block(concat4, num_filters*8)

    up3 = tf.keras.layers.UpSampling2D(2)(deconv4)
    attn3 = attention_gate(conv3, up3, num_filters*2)
    concat3 = tf.keras.layers.Concatenate()([up3, attn3])
    deconv3 = residual_block(concat3, num_filters*4)

    up2 = tf.keras.layers.UpSampling2D(2)(deconv3)
    attn2 = attention_gate(conv2, up2, num_filters)
    concat2 = tf.keras.layers.Concatenate()([up2, attn2])
    deconv2 = residual_block(concat2, num_filters*2)

    up1 = tf.keras.layers.UpSampling2D(2)(deconv2)
    attn1 = attention_gate(conv1, up1, num_filters//2)
    concat1 = tf.keras.layers.Concatenate()([up1, attn1])
    deconv1 = residual_block(concat1, num_filters)

    output_conv = tf.keras.layers.Conv2D(3, 1, activation='sigmoid', padding='same')(deconv1)

    resized_inputs = tf.keras.layers.Lambda(
        lambda x: tf.image.resize(x[0], tf.shape(x[1])[1:3])
    )([inputs, output_conv])

    outputs = tf.keras.layers.Add()([output_conv, resized_inputs])
    outputs = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0, 1))(outputs)

    return tf.keras.Model(inputs, outputs)

# --- Load Model from Google Drive ---
@st.cache_resource
def load_model():
    model_path = "denoising_model.keras"
    
    # Load the model directly from your local folder
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'combined_loss': combined_loss,
            'psnr_metric': psnr_metric,
            'ssim_metric': ssim_metric
        }
    )
    return model

# --- Enhance Image ---
def enhance_image(model, image_array, original_size):
    input_tensor = np.expand_dims(image_array, axis=0)
    enhanced_array = model.predict(input_tensor)[0]
    enhanced_array = np.clip(enhanced_array, 0, 1) * 255
    enhanced_img = Image.fromarray(enhanced_array.astype(np.uint8))
    return enhanced_img.resize(original_size, Image.LANCZOS)

# --- Main App ---
def main():
    tab1, tab2 = st.tabs(["Image Enhancement", "About"])

    with tab1:
        st.markdown("<h2 class='sub-header'>Upload Your Image</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a low-light image...", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h3>Original Image</h3>", unsafe_allow_html=True)
                st.image(image, use_container_width=True)

            if st.button("Enhance Image"):
                with st.spinner("Enhancing your image..."):
                    model = load_model()
                    processed_img, original_size = preprocess_image(image)
                    enhanced_img = enhance_image(model, processed_img, original_size)
                    with col2:
                        st.markdown("<h3>Enhanced Image</h3>", unsafe_allow_html=True)
                        st.image(enhanced_img, use_container_width=True)
                    buf = io.BytesIO()
                    enhanced_img.save(buf, format="PNG")
                    st.download_button("Download Enhanced Image", buf.getvalue(), "enhanced_image.png", "image/png")

    with tab2:
        st.markdown("<h2 class='sub-header'>About this App</h2>", unsafe_allow_html=True)
        st.write("""
        This application uses a deep learning model built on U-Net with residual blocks and attention mechanisms to enhance low-light images.
        Features:
        - Convolutional Block Attention Modules (CBAM)
        - Attention Gates
        - Combined SSIM and MSE loss
        """)

if __name__ == "__main__":
    main()