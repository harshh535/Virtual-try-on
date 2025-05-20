
import streamlit as st
import pyrebase
import base64
from io import BytesIO
from PIL import Image

def main():
    if not st.session_state.get('user_logged_in'):
        st.error("🔒 Please login through the main portal first")
        st.experimental_rerun()  # Redirect back to auth

    with open('user_id.jpg', 'rb') as img_file:
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
                    url('data:image/jpeg;base64,{encoded_bg}') no-repeat center/cover !important;
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

    # Add custom CSS for top-left positioning and transparent style
    st.markdown('''
        <style>
        .logout-btn-fixed {
            position: fixed;
            left: 20px;
            top: 20px;
            z-index: 9999;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #fff !important;
            font-size: 1.1rem !important;
        }
        </style>
    ''', unsafe_allow_html=True)
    logout_placeholder = st.empty()
    logout_html = '''
        <form action="#" method="post">
            <button class="logout-btn-fixed" type="submit">🔓 Logout</button>
        </form>
    '''
    logout_placeholder.markdown(logout_html, unsafe_allow_html=True)

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

    firebase = pyrebase.initialize_app(firebase_config)
    db = firebase.database()

    def get_shop_names():
        try:
            shops = db.child("Shops").get().val()
            return list(shops.keys()) if shops else []
        except Exception as e:
            st.error(f"⚠️ Error fetching shop names: {e}")
            return []

    def get_shop_items(shop_name):
        try:
            items = db.child(f"Shops/{shop_name}/items").get().val()
            return items if items else {}
        except Exception as e:
            st.error(f"⚠️ Error fetching items: {e}")
            return {}

    def decode_image(base64_string):
        try:
            img_bytes = base64.b64decode(base64_string)
            img = BytesIO(img_bytes)
            return Image.open(img)
        except Exception as e:
            st.error(f"⚠️ Error decoding image: {e}")
            return None

    def add_to_favorites(user_id, shop_name, item_id, item_data):
        try:
            fav_path = f"Users/{user_id}/favorites/{shop_name}/{item_id}"
            db.child(fav_path).set(item_data)
        except Exception as e:
            st.error(f"⚠️ Failed to add to favorites: {e}")

    def remove_from_favorites(user_id, shop_name, item_id):
        try:
            fav_path = f"Users/{user_id}/favorites/{shop_name}/{item_id}"
            db.child(fav_path).remove()
        except Exception as e:
            st.error(f"⚠️ Failed to remove from favorites: {e}")

    def get_favorites(user_id):
        try:
            favorites = db.child(f"Users/{user_id}/favorites").get().val()
            return favorites if favorites else {}
        except Exception as e:
            st.error(f"⚠️ Error fetching favorites: {e}")
            return {}

    if "page" not in st.session_state:
        st.session_state["page"] = "home"
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = "User1"

    if st.session_state["page"] == "home":
        st.title("🏪 Select a Shop")
        col1, col2 = st.columns([4, 1])
        with col1:
            shop_names = get_shop_names()
            selected_shop = st.selectbox("Choose a shop:", shop_names)
        with col2:
            if st.button("⭐ Favorites", key="view_favorites"):
                st.session_state["page"] = "favorites"
                st.rerun()
        if st.button("Explore Shop") and selected_shop:
            st.session_state["selected_shop"] = selected_shop.strip()
            st.session_state["page"] = "inventory"
            st.rerun()

    elif st.session_state["page"] == "favorites":
        st.title("⭐ Your Favorites")
        user_id = st.session_state["user_id"]
        favorites = get_favorites(user_id)
        if favorites:
            for shop_name, items in favorites.items():
                for item_id, item_data in items.items():
                    col1, col2 = st.columns([2, 8])
                    with col1:
                        st.write(f"🏪 **{shop_name}**")
                    with col2:
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
                                img = decode_image(item_data["original_image"])
                                if img:
                                    st.image(img, width=250, caption="Original Image")
                            overlay_keys = [key for key in item_data.keys() if key.startswith("overlayed_")]
                            if overlay_keys:
                                st.subheader("👕 Try-On Results")
                                cols = st.columns(3)
                                for i, key in enumerate(overlay_keys):
                                    img = decode_image(item_data[key])
                                    if img:
                                        with cols[i % 3]:
                                            st.image(img, width=250, caption=f"Model {i+1}", use_container_width=True)
                            else:
                                st.warning("⚠️ No overlayed images available.")
                            # Add a right-aligned cross button inside the expander
                            btn_cols = st.columns([8, 1])
                            with btn_cols[1]:
                                if st.button("❌", key=f"remove_{shop_name}_{item_id}"):
                                    remove_from_favorites(user_id, shop_name, item_id)
                                    st.rerun()
        else:
            st.warning("⚠️ No items in favorites yet.")
        if st.button("🔙 Back to Shop Selection"):
            st.session_state["page"] = "home"
            st.rerun()

    elif st.session_state["page"] == "inventory":
        selected_shop = st.session_state.get("selected_shop", None)
        if not selected_shop:
            st.error("⚠️ No shop selected. Please go back and select a shop.")
            st.session_state["page"] = "home"
            st.rerun()
        st.title(f"🛒 Inventory of {selected_shop}")
        items = get_shop_items(selected_shop)
        user_id = st.session_state["user_id"]
        favorites = get_favorites(user_id)
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
                            img = decode_image(item_data["original_image"])
                            if img:
                                st.image(img, width=250, caption="Original Image")
                        overlay_keys = [key for key in item_data.keys() if key.startswith("overlayed_")]
                        if overlay_keys:
                            st.subheader("👕 Try-On Results")
                            cols = st.columns(3)
                            for i, key in enumerate(overlay_keys):
                                img = decode_image(item_data[key])
                                if img:
                                    with cols[i % 3]:
                                        st.image(img, width=250, caption=f"Model {i+1}", use_container_width=True)
                with col2:
                    if selected_shop in favorites and item_id in favorites[selected_shop]:
                        if st.button("❌", key=f"remove_{item_id}"):
                            remove_from_favorites(user_id, selected_shop, item_id)
                            st.rerun()
                    else:
                        if st.button("⭐", key=f"fav_{item_id}"):
                            add_to_favorites(user_id, selected_shop, item_id, item_data)
                            st.rerun()
        else:
            st.warning("⚠️ No items available in this shop.")
        if st.button("🔙 Back to Shop Selection"):
            st.session_state["page"] = "home"
            st.rerun()

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