import streamlit as st
import pyrebase
import shopkeeper_dashboard
import user_dashboard
import os
import base64

# ✅ Streamlit Page Config (must be first)
st.set_page_config(page_title="Virtual Try-On", page_icon="👕", layout="wide")

# --- Load and encode background image (relative path) ---
with open('mull.jpg', 'rb') as img_file:
    img_bytes = img_file.read()
    encoded_bg = base64.b64encode(img_bytes).decode()

# ✅ Custom CSS Styling: simple background image with overlay
st.markdown(f"""
<style>
/* Full-page background with fade animation */
.stApp {{
    background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                url("data:image/jpeg;base64,{encoded_bg}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    color: #FFFFFF;
    animation: bgFade 8s ease-in-out infinite;
}}

@keyframes bgFade {{
    0%, 100% {{ backdrop-filter: brightness(0.95); }}
    50% {{ backdrop-filter: brightness(1.25); }}
}}

/* Hero headline with gradient */
.hero-text {{
    font-size: 3rem;
    text-align: center;
    margin: 2rem 0;
    background: linear-gradient(90deg, #ff8a00, #e52e71);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-size: 200% auto;
    animation: shine 4s linear infinite;
    position: relative;
    z-index: 2;
    padding: 20px;
    width: 100%;
    display: block;
}}

@keyframes shine {{
    to {{
        background-position: 200% center;
    }}
}}

/* Make sure the hero text is visible */
h1.hero-text {{
    color: #FFFFFF;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    font-weight: bold;
    margin-top: 50px;
    margin-bottom: 50px;
}}

/* Role selection buttons */
.stButton > button {{
    font-size: 20px !important;
    padding: 14px 28px !important;
    margin: 0 1rem;
    border-radius: 8px !important;
    background-color: rgba(255,255,255,0.1);
    color: #FFFFFF;
    border: 2px solid rgba(255,255,255,0.6);
    position: relative;
    display: inline-block;
}}

/* Styles for buttons within the centered container */
.button-container .stButton > button {{
    display: inline-block !important;
    margin: 0 !important;
    width: fit-content !important;
}}

/* Content styling with fade-in effect */
.content {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    max-width: 800px;
    margin: 0 auto;
    text-align: center;
}}

.content p {{
    font-size: 1.4rem;
    line-height: 1.8;
    color: #DDDDDD;
    margin-bottom: 2rem;
    animation: fadeIn 1.5s ease-in;
    text-align: center;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
}}

/* Button container */
.button-container {{
    position: relative;
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 2rem;
    margin: 2rem 0;
    height: auto;
}}

/* Center the role selection buttons */
[data-testid="column"] {{
    display: flex;
    justify-content: center;
    align-items: center;
}}

/* Center the main title */
h1, h2, h3 {{
    text-align: center;
    width: 100%;
}}

@keyframes fadeIn {{
    0% {{ opacity: 0; }}
    100% {{ opacity: 1; }}
}}

/* Remove extra top padding */
.css-12w0qpk {{ padding-top: 0; }}
</style>
""", unsafe_allow_html=True)

# ✅ Hero Section (text only)
st.markdown(
    """
    <h1 class='hero-text'>Empowering Local Shopkeepers, Connecting Communities</h1>
    """,
    unsafe_allow_html=True
)

# --- Firebase Setup ---
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
auth = firebase.auth()
db = firebase.database()

# --- Session State Initialization ---
if 'role_selected' not in st.session_state:
    st.session_state.role_selected = None
if 'shopkeeper_logged_in' not in st.session_state:
    st.session_state.shopkeeper_logged_in = False
if 'user_logged_in' not in st.session_state:
    st.session_state.user_logged_in = False
if 'shop_no' not in st.session_state:
    st.session_state.shop_no = ''
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# --- Auth Helper Functions ---
def register_shopkeeper(email, password, owner_name, phone):
    try:
        shops = db.child("Shops").get().val() or {}
        shop_no = f"Shop No {len(shops) + 1}"
        db.child("Shops").child(shop_no).set({
            "email": email,
            "password": password,
            "owner_name": owner_name,
            "phone_number": phone
        })
        st.success(f"✅ Registered! Your shop ID: {shop_no}")
    except Exception as e:
        st.error(f"⚠️ Registration Error: {e}")

def login_shopkeeper(email, password):
    try:
        shops = db.child("Shops").get().val() or {}
        for shop_no, info in shops.items():
            if info.get("email") == email and info.get("password") == password:
                st.session_state.shopkeeper_logged_in = True
                st.session_state.shop_no = shop_no
                st.session_state.role_selected = 'shopkeeper'
                st.success(f"✅ Welcome back, {shop_no}!")
                st.rerun()
        st.error("⚠️ Invalid shopkeeper credentials.")
    except Exception as e:
        st.error(f"⚠️ Login Error: {e}")

def register_user(email, password, full_name, phone):
    try:
        users = db.child("Users").get().val() or {}
        user_id = f"User {len(users) + 1}"
        db.child("Users").child(user_id).set({
            "email": email,
            "password": password,
            "full_name": full_name,
            "phone_number": phone
        })
        st.success("✅ User registered successfully!")
    except Exception as e:
        st.error(f"⚠️ Registration Error: {e}")

def login_user(email, password):
    try:
        users = db.child("Users").get().val() or {}
        for uid, info in users.items():
            if info.get("email") == email and info.get("password") == password:
                st.session_state.user_logged_in = True
                st.session_state.user_id = uid
                st.session_state.role_selected = 'user'
                st.success(f"✅ Welcome, {info.get('full_name', uid)}!")
                st.rerun()
                return
        st.error("⚠️ Invalid user credentials.")
    except Exception as e:
        st.error(f"⚠️ Login Error: {e}")

# --- Main Flow ---
if st.session_state.role_selected is None:
    st.markdown("## Choose Your Path", unsafe_allow_html=True)

    # Use columns to create a centered area for the buttons
    col_left, col_center, col_right = st.columns([1, 2, 1]) # Adjust ratios as needed for centering

    with col_center:
        # Use a container within the centered column for the buttons
        with st.container():
            # Apply CSS class to the container for potential flexbox centering
            st.markdown('<div class="button-container">', unsafe_allow_html=True)

            # Use columns to place buttons side-by-side
            button_col1, button_col2 = st.columns(2)

            with button_col1:
                # Place buttons directly inside the container with their logic
                shopkeeper_btn = st.button("👔 Shopkeeper Portal", key="shopkeeper_btn")
            with button_col2:
                customer_btn = st.button("🛍 Customer Portal", key="customer_btn")

            st.markdown('</div>', unsafe_allow_html=True)

            # Process button clicks
            if shopkeeper_btn:
                st.session_state.role_selected = 'shopkeeper'
                st.rerun()
            if customer_btn:
                st.session_state.role_selected = 'user'
                st.rerun()

    # Text Content Below Buttons (centered)
    st.markdown("""
    <div class='content'>
      <p><strong>Virtual Try-On</strong> empowers local businesses by giving them an online storefront and helps customers discover unique fashion from their communities.</p>
      
      <p>Browse boutique collections, <em>virtually try on</em> outfits for confidence, and seamlessly purchase items—all from your device.</p>
      
      <p>Together, we build stronger local economies, foster personal connections, and make every shopping experience unforgettable.</p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.role_selected == 'shopkeeper':
    if st.session_state.shopkeeper_logged_in:
        shopkeeper_dashboard.show_dashboard()
    else:
        st.subheader('🔑 Shopkeeper Login / Register')
        tabs = st.tabs(['Login', 'Register'])
        with tabs[0]:
            email = st.text_input('Email', key='sk_email')
            pwd = st.text_input('Password', type='password', key='sk_pwd')
            if st.button('Login', key='sk_login'):
                login_shopkeeper(email, pwd)
        with tabs[1]:
            re_email = st.text_input('Email', key='sk_reg_email')
            re_pwd = st.text_input('Password', type='password', key='sk_reg_pwd')
            owner = st.text_input('Owner Name', key='sk_owner')
            phone = st.text_input('Phone Number', key='sk_phone')
            if st.button('Register', key='sk_register'):
                register_shopkeeper(re_email, re_pwd, owner, phone)

elif st.session_state.role_selected == 'user':
    if st.session_state.user_logged_in:
        try:
            user_dashboard.main()
        except Exception as e:
            st.error(f"⚠️ Error loading user dashboard: {e}")
            st.button("🔙 Back to Main", on_click=lambda: setattr(st.session_state, 'role_selected', None))
    else:
        st.subheader('🔑 Customer Login / Register')
        tabs = st.tabs(['Login', 'Register'])
        with tabs[0]:
            ue = st.text_input('Email', key='user_email')
            up = st.text_input('Password', type='password', key='user_pwd')
            if st.button('Login', key='user_login'):
                login_user(ue, up)
        with tabs[1]:
            ur_email = st.text_input('Email', key='user_reg_email')
            ur_pwd = st.text_input('Password', type='password', key='user_reg_pwd')
            name = st.text_input('Full Name', key='user_name')
            uphone = st.text_input('Phone Number', key='user_phone')
            if st.button('Register', key='user_register'):
                register_user(ur_email, ur_pwd, name, uphone)