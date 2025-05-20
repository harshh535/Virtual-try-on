import streamlit as st
import pyrebase
import base64
import os
import subprocess
import time
from io import BytesIO

# ✅ Firebase Configuration
firebase_config = {
    "apiKey": "AIzaSyAm6HtleJzgdBAMM7k0VGaFwfQbe_GNbWY",
    "authDomain": "clothes-wala.firebaseapp.com",
    "databaseURL": "https://clothes-wala-default-rtdb.firebaseio.com",
    "projectId": "clothes-wala",
    "storageBucket": "clothes-wala.appspot.com",
    "messagingSenderId": "48604679015",
    "appId": "1:48604679015:web:f0065e957c1b50eb563dc2",
    "measurementId": "G-HMDVX52K6Q"
}

# ✅ Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
db = firebase.database()

def run_virtual_tryon(cloth_path):
    """Runs the virtual try-on backend script."""
    try:
        subprocess.run(["python", "automated.py", cloth_path])

        return True
    except Exception as e:
        st.error(f"⚠️ Error running virtual try-on: {e}")
        return False

def encode_image(image_path):
    """Encodes an image to Base64 format."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        st.error(f"⚠️ Error encoding image: {e}")
        return None

def upload_item(shop_no, item_name, item_price, item_desc, uploaded_image):
    """Uploads item details and overlayed images to Firebase."""
    if uploaded_image is not None:
        try:
            item_id = f"item_{int(time.time())}"  # Unique ID for item
            item_path = f"Shops/{shop_no}/items/{item_id}/"
            
            # Convert uploaded image to Base64
            image_bytes = uploaded_image.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            # Save details to Firebase
            db.child(item_path).set({
                "name": item_name,
                "price": item_price,
                "description": item_desc,
                "original_image": image_base64
            })
            
            # Save image locally for processing
            cloth_folder = "datasets/test/cloth/"
            os.makedirs(cloth_folder, exist_ok=True)
            cloth_path = os.path.join(cloth_folder, uploaded_image.name)
            with open(cloth_path, "wb") as f:
                f.write(image_bytes)
            
            # Run virtual try-on process
            if run_virtual_tryon(cloth_path):
                # Upload overlayed images
                results_folder = "results/"
                if os.path.exists(results_folder):
                    overlayed_images = [os.path.join(results_folder, img) for img in os.listdir(results_folder) if img.endswith((".jpg", ".png"))]
                    overlayed_images_base64 = {f"overlayed_{i}": encode_image(img) for i, img in enumerate(overlayed_images)}
                    db.child(item_path).update(overlayed_images_base64)
                
                st.success("✅ Item uploaded successfully with overlayed images!")
            else:
                st.error("⚠️ Failed to generate try-on images.")
        except Exception as e:
            st.error(f"⚠️ Error uploading item: {e}")
    else:
        st.error("⚠️ Please upload an image first.")

def get_shop_items(shop_no):
    """Fetches all items of a shop from Firebase."""
    try:
        items = db.child(f"Shops/{shop_no}/items").get().val()
        return items if items else {}
    except Exception as e:
        st.error(f"⚠️ Error fetching items: {e}")
        return {}

def show_dashboard():
    """Displays the Shopkeeper Dashboard."""
    if not st.session_state.get('shopkeeper_logged_in'):
        st.error("🔒 Please login through the main portal first")
        st.rerun()

    with open('ma.jpg', 'rb') as img_file:
        img_bytes = img_file.read()
        encoded_bg = base64.b64encode(img_bytes).decode()

    st.markdown(f"""
    <style>
    @keyframes boxFade {{
        0%, 100% {{ 
            background-color: rgba(44, 44, 44, 0.95);
            border-color: rgba(64, 64, 64, 0.95);
        }}
        50% {{ 
            background-color: rgba(44, 44, 44, 0.2);
            border-color: rgba(64, 64, 64, 0.2);
        }}
    }}

    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                    url('data:image/png;base64,{encoded_bg}') no-repeat center/cover !important;
        background-size: cover !important;
    }}

    /* Animated grey box styling for items */
    div[data-testid="stExpander"] {{
        position: relative;
        border-radius: 8px !important;
        padding: 15px !important;
        margin-bottom: 15px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        background-color: rgba(44, 44, 44, 0.95);
        animation: boxFade 5s ease-in-out infinite;
        transition: all 0.3s ease;
    }}

    /* Style for expander header */
    div[data-testid="stExpander"] > div:first-child {{
        color: white !important;
        background-color: transparent !important;
        z-index: 2;
        position: relative;
    }}

    /* Style for expander content */
    div[data-testid="stExpander"] > div:last-child {{
        color: white !important;
        background-color: transparent !important;
        z-index: 2;
        position: relative;
    }}

    /* Make all text white and ensure it stays on top */
    div[data-testid="stExpander"] p,
    div[data-testid="stExpander"] span,
    div[data-testid="stExpander"] strong {{
        color: #FFFFFF !important;
        position: relative;
        z-index: 2;
    }}

    /* Style for buttons inside expander */
    div[data-testid="stExpander"] button {{
        background-color: #404040 !important;
        color: white !important;
        border: 1px solid #505050 !important;
        position: relative;
        z-index: 2;
    }}
    </style>
    """, unsafe_allow_html=True)

    st.title(f"🏪 {st.session_state.shop_no} - Shopkeeper Dashboard")
    
    # ✅ Upload Section
    st.subheader("Upload New Item")
    
    # Input fields one per line
    item_name = st.text_input("Item Name")
    item_price = st.number_input("Price", min_value=0.0, format="%.2f")
    item_color = st.text_input("Color")
    item_type = st.text_input("Type")
    item_length = st.text_input("Length")
    uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    
    if st.button("Upload Item", use_container_width=True):
        # Validate all fields are filled
        if not item_name:
            st.error("⚠️ Please enter item name")
        elif item_price <= 0:
            st.error("⚠️ Please enter a valid price")
        elif not uploaded_image:
            st.error("⚠️ Please upload an image")
        else:
            item_desc = f"Color: {item_color} | Type: {item_type} | Length: {item_length}"
            upload_item(st.session_state.shop_no, item_name, item_price, item_desc, uploaded_image)
    
    # Add yellow line note
    st.markdown("""
    <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        <p style='color: #856404; margin: 0;'>⚠️ Note: Please take photos with a white background for best results. After uploading an item, the virtual try-on process will start automatically. This may take a few moments.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ✅ Display Inventory
    st.subheader("🛒 Shop Inventory")
    items = get_shop_items(st.session_state.shop_no)
    
    if items:
        for item_id, item_data in items.items():
            col1, col2 = st.columns([8, 2])
            with col1:
                with st.expander(f"🔹 {item_data.get('name', 'Unnamed Item')}"):
                    desc = item_data.get('description', '')
                    color, type_, length = '', '', ''
                    if desc and 'Color:' in desc and 'Type:' in desc and 'Length:' in desc:
                        parts = dict(part.strip().split(':', 1) for part in desc.split('|') if ':' in part)
                        color = parts.get('Color', '').strip()
                        type_ = parts.get('Type', '').strip()
                        length = parts.get('Length', '').strip()
                    
                    st.markdown(f"""
                        <span style='font-size:1.5rem; font-weight:bold;'>
                            <span style='margin-right:60px;'>💰 Price: {item_data.get('price', 'N/A')}</span>
                            <span style='margin-right:60px;'>🎨 Color: {color}</span>
                            <span style='margin-right:60px;'>👕 Type: {type_}</span>
                            <span>📏 Length: {length}</span>
                        </span>
                    """, unsafe_allow_html=True)
                    
                    if "original_image" in item_data:
                        img_bytes = base64.b64decode(item_data["original_image"])
                        img = BytesIO(img_bytes)
                        st.image(img, width=250, caption="Original Image")
                    
                    overlay_keys = [key for key in item_data.keys() if key.startswith("overlayed_")]
                    if overlay_keys:
                        st.subheader("👕 Try-On Results")
                        cols = st.columns(3)
                        for i, key in enumerate(overlay_keys):
                            img_bytes = base64.b64decode(item_data[key])
                            img = BytesIO(img_bytes)
                            with cols[i % 3]:
                                st.image(img, width=250, caption=f"Model {i+1}", use_container_width=True)
                    else:
                        st.warning("⚠️ No overlayed images available.")
            with col2:
                if st.button("🗑️", key=f"delete_{item_id}"):
                    try:
                        db.child(f"Shops/{st.session_state.shop_no}/items/{item_id}").remove()
                        st.success(f"✅ {item_data.get('name', 'Item')} deleted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"⚠️ Error deleting item: {e}")
    else:
        st.warning("⚠️ No items found in the shop.")
    
    # Add extra space before Logout button to avoid overlap
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()

    st.write("")
    st.write("")

# ✅ Run the Dashboard when this script is executed
if __name__ == "__main__":
    show_dashboard() 
