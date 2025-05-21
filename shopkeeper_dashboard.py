import streamlit as st
import pyrebase
import base64
import os
import asyncio
import time
from io import BytesIO
import sys
import signal

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

async def run_virtual_tryon(cloth_path):
    """Run automated.py asynchronously with timeout handling"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        automated_path = os.path.join(base_dir, "automated.py")
        
        # Create required directories
        required_dirs = [
            os.path.join(base_dir, "results"),
            os.path.join(base_dir, "datasets", "test", "cloth"),
            os.path.join(base_dir, "datasets", "test", "cloth-mask")
        ]
        for d in required_dirs:
            os.makedirs(d, exist_ok=True)

        process = await asyncio.create_subprocess_exec(
            sys.executable, automated_path, cloth_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=base_dir
        )
        
        # 15-minute timeout (Streamlit Cloud's limit)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=900)
            if process.returncode != 0:
                st.error(f"❌ Try-on failed: {stderr.decode()}")
                return False
            return True
        except asyncio.TimeoutError:
            st.error("🕒 Process timed out after 15 minutes")
            process.kill()
            return False
            
    except Exception as e:
        st.error(f"🚨 Critical error: {str(e)}")
        return False

def encode_image(image_path):
    """Encodes an image to Base64 format with validation"""
    try:
        if not os.path.exists(image_path):
            st.error(f"⚠️ Image not found: {image_path}")
            return None
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        st.error(f"⚠️ Encoding failed: {e}")
        return None

def upload_item(shop_no, item_name, item_price, item_desc, uploaded_image):
    if uploaded_image is not None:
        try:
            item_id = f"item_{int(time.time())}"
            item_path = f"Shops/{shop_no}/items/{item_id}/"
            base_dir = os.path.dirname(os.path.abspath(__file__))

            # Save image locally with unique name
            cloth_folder = os.path.join(base_dir, "datasets", "test", "cloth")
            os.makedirs(cloth_folder, exist_ok=True)
            unique_name = f"{int(time.time())}_{uploaded_image.name}"
            cloth_path = os.path.join(cloth_folder, unique_name)
            
            with open(cloth_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

            # Save initial data to Firebase
            db.child(item_path).set({
                "name": item_name,
                "price": item_price,
                "description": item_desc,
                "original_image": base64.b64encode(uploaded_image.getvalue()).decode("utf-8")
            })

            # Run try-on process
            with st.spinner("🧪 Generating virtual try-ons (this may take 2-5 minutes)..."):
                success = asyncio.run(run_virtual_tryon(cloth_path))

                if success:
                    results_dir = os.path.join(base_dir, "results")
                    overlayed_images = []
                    if os.path.exists(results_dir):
                        overlayed_images = [os.path.join(results_dir, f) 
                                          for f in os.listdir(results_dir) 
                                          if f.lower().endswith((".jpg", ".png"))]

                    if overlayed_images:
                        updates = {}
                        for i, img_path in enumerate(overlayed_images):
                            if encoded := encode_image(img_path):
                                updates[f"overlayed_{i}"] = encoded
                            else:
                                st.warning(f"⚠️ Failed to encode image: {os.path.basename(img_path)}")
                        
                        if updates:
                            db.child(item_path).update(updates)
                            st.success("✅ Item uploaded with try-on results!")
                            st.balloons()
                        else:
                            st.error("⚠️ All try-on results failed to encode")
                    else:
                        st.warning("⚠️ Virtual try-on completed but no images generated")
                else:
                    st.error("❌ Failed to process virtual try-on")

        except Exception as e:
            st.error(f"⚠️ Upload failed: {e}")
            if os.path.exists(cloth_path):
                os.remove(cloth_path)
    else:
        st.error("⚠️ Please upload an image first")

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
