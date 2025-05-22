import streamlit as st
import pyrebase
import base64
import os
import sys
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

# Determine the base directory of this script
base_dir = os.path.dirname(os.path.abspath(__file__))

# ✅ Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
db = firebase.database()

def run_virtual_tryon(cloth_path):
    """
    Runs the virtual try-on pipeline by calling `automated.main()` directly
    instead of spawning a subprocess. Assumes `automated.py` lives alongside this file.
    """
    try:
        # Import the module containing the in-process pipeline
        import automated

        st.text(f"▶️ Running automated.main() on cloth_path:\n   {cloth_path}\n")
        # Call the main() function from automated.py
        automated.main(cloth_path)

        st.success("✅ Virtual try-on pipeline completed successfully.")
        return True
    except Exception as e:
        st.error(f"⚠️ Error running virtual try-on: {e}")
        import traceback
        st.error(traceback.format_exc())
        return False

def encode_image(image_path):
    """Encodes an image file to a Base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        st.error(f"⚠️ Error encoding image: {e}")
        return None

def upload_item(shop_no, item_name, item_price, item_desc, uploaded_image):
    """
    Saves the uploaded cloth image, runs the try-on pipeline via `automated.main()`,
    and writes original + overlay results to Firebase under Shops/{shop_no}/items/{item_id}/.
    """
    if uploaded_image is None:
        st.error("⚠️ Please upload an image first.")
        return

    try:
        # 1) Create unique item ID and Firebase path
        item_id = f"item_{int(time.time())}"
        item_path = f"Shops/{shop_no}/items/{item_id}/"

        # 2) Convert uploaded image to Base64 and store metadata first
        image_bytes = uploaded_image.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        db.child(item_path).set({
            "name": item_name,
            "price": item_price,
            "description": item_desc,
            "original_image": image_base64
        })

        # 3) Save cloth image into datasets/test/cloth/
        cloth_folder = os.path.join(base_dir, "datasets", "test", "cloth")
        os.makedirs(cloth_folder, exist_ok=True)
        cloth_path = os.path.join(cloth_folder, uploaded_image.name)
        with open(cloth_path, "wb") as f:
            f.write(image_bytes)

        # 4) Run the try-on pipeline in-process
        with st.spinner("🧪 Generating virtual try-ons (this may take a few minutes)..."):
            success = run_virtual_tryon(cloth_path)

        if not success:
            st.error("❌ Failed to generate try-on images.")
            return

        # 5) Read overlayed images from results/ and upload Base64 to Firebase
        results_folder = os.path.join(base_dir, "results")
        if os.path.exists(results_folder):
            overlayed_images = [
                os.path.join(results_folder, fn)
                for fn in os.listdir(results_folder)
                if fn.lower().endswith((".jpg", ".png"))
            ]
            if overlayed_images:
                updates = {}
                for i, img_path in enumerate(sorted(overlayed_images)):
                    encoded = encode_image(img_path)
                    if encoded:
                        updates[f"overlayed_{i}"] = encoded
                    else:
                        st.warning(f"⚠️ Failed to encode overlay image: {os.path.basename(img_path)}")
                if updates:
                    db.child(item_path).update(updates)
                    st.success("✅ Item uploaded successfully with overlayed images!")
                    st.balloons()
                else:
                    st.error("⚠️ All overlayed images failed to encode.")
            else:
                st.warning("⚠️ No overlayed images found in results/.")
        else:
            st.warning("⚠️ 'results/' folder does not exist after pipeline.")

    except Exception as e:
        st.error(f"⚠️ Error uploading item: {e}")
        import traceback
        st.error(traceback.format_exc())

def get_shop_items(shop_no):
    """Fetches all items under a shop from Firebase."""
    try:
        items = db.child(f"Shops/{shop_no}/items").get().val()
        return items or {}
    except Exception as e:
        st.error(f"⚠️ Error fetching items: {e}")
        return {}

def show_dashboard():
    """Displays the shopkeeper dashboard: upload form + inventory list."""
    if not st.session_state.get("shopkeeper_logged_in"):
        st.error("🔒 Please login through the main portal first.")
        st.rerun()

    # Load background image (ma.jpg) and set CSS
    try:
        with open(os.path.join(base_dir, "ma.jpg"), "rb") as img_file:
            encoded_bg = base64.b64encode(img_file.read()).decode()
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

        div[data-testid="stExpander"] > div:first-child {{
            color: white !important;
            background-color: transparent !important;
            z-index: 2;
            position: relative;
        }}

        div[data-testid="stExpander"] > div:last-child {{
            color: white !important;
            background-color: transparent !important;
            z-index: 2;
            position: relative;
        }}

        div[data-testid="stExpander"] p,
        div[data-testid="stExpander"] span,
        div[data-testid="stExpander"] strong {{
            color: #FFFFFF !important;
            position: relative;
            z-index: 2;
        }}

        div[data-testid="stExpander"] button {{
            background-color: #404040 !important;
            color: white !important;
            border: 1px solid #505050 !important;
            position: relative;
            z-index: 2;
        }}
        </style>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        # If ma.jpg is missing, skip background styling
        pass

    st.title(f"🏪 {st.session_state.shop_no} - Shopkeeper Dashboard")

    # ================= Upload Section =================
    st.subheader("Upload New Item")
    item_name = st.text_input("Item Name")
    item_price = st.number_input("Price", min_value=0.0, format="%.2f")
    item_color = st.selectbox("Color", ["Red", "Blue", "Green", "Black", "White", "Yellow", "Other"])
    item_type = st.selectbox("Type", ["T-Shirt", "Kurti", "Jacket", "Shirt", "Dress", "Other"])
    item_length = st.selectbox("Length", ["Short", "Medium", "Long"])
    uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if st.button("Upload Item", use_container_width=True):
        if not item_name:
            st.error("⚠️ Please enter item name.")
        elif item_price <= 0:
            st.error("⚠️ Please enter a valid price.")
        elif not uploaded_image:
            st.error("⚠️ Please upload an image.")
        else:
            item_desc = f"Color: {item_color} | Type: {item_type} | Length: {item_length}"
            upload_item(
                st.session_state.shop_no,
                item_name,
                item_price,
                item_desc,
                uploaded_image
            )

    st.markdown("""
    <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        <p style='color: #856404; margin: 0;'>
            ⚠️ Note: Please take photos with a white background for best results.
            After uploading an item, the virtual try-on process will start automatically.
            This may take a few moments.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ================= Display Inventory =================
    st.subheader("🛒 Shop Inventory")
    items = get_shop_items(st.session_state.shop_no)

    if items:
        for item_id, item_data in items.items():
            col1, col2 = st.columns([8, 2])
            with col1:
                with st.expander(f"🔹 {item_data.get('name', 'Unnamed Item')}"):
                    desc = item_data.get("description", "")
                    color, type_, length = "", "", ""
                    if desc and "Color:" in desc and "Type:" in desc and "Length:" in desc:
                        try:
                            parts = {
                                part.strip().split(":", 1)[0]: part.strip().split(":", 1)[1]
                                for part in desc.split("|")
                                if ":" in part
                            }
                            color = parts.get("Color", "").strip()
                            type_ = parts.get("Type", "").strip()
                            length = parts.get("Length", "").strip()
                        except:
                            color = type_ = length = ""

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
                        for i, key in enumerate(sorted(overlay_keys, key=lambda k: int(k.split("_")[1]))):
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

    # Extra spacing before Logout
    st.write("\n" * 5)
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()

# ✅ Run the Dashboard when this script is executed
if __name__ == "__main__":
    show_dashboard()

