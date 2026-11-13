import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Hotel Cancellation Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Global Custom Styling
st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }
    .hero-container {
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        border: 1px solid #374151;
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #F9FAFB;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #9CA3AF;
        margin-bottom: 0;
    }
    section[data-testid="stSidebar"] {
        background-color: #0D1117;
        border-right: 1px solid #21262D;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# 3. Main Header Banner
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">Hotel Cancellation Risk Intelligence System</div>
        <div class="hero-subtitle">
            An end-to-end Machine Learning suite designed to predict reservation cancellations, assess revenue risk, and deliver data-driven operational decision strategies.
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

# 4. Executive Overview & Business Objectives
st.header("Executive Overview & Project Goals")

col1, col2 = st.columns(2)

with col1:
  st.subheader("Business Context & Challenge")
  st.write("""
    Online reservation channels have revolutionized booking patterns, but free or low-cost cancellation options leave hotels exposed to sudden revenue drop-offs, inaccurate demand forecasting, and inefficient staff scheduling.
    """)

with col2:
  st.subheader("Key Operational Objectives")
  st.markdown("""
    1. **Predictive Cancellation Risks:** Detect reservations likely to call off before arrival using historical guest behavior.
    2. **Actionable Revenue Management:** Implement targeted non-refundable policies and deposit requirements for high-risk bookings.
    3. **Optimized Resource Allocation:** Tailor room allocations and staffing schedules based on lead times and guest commitment signals.
    """)

st.markdown("---")
st.sidebar.text("Select a module from the left menu to begin.")