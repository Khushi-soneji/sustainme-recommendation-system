import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SustainMe ♻️",
    page_icon="♻️",
    layout="centered"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS — Earthy organic theme
#  Deep forest green + warm cream + clay orange
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --forest:   #1a3a2a;
    --moss:     #2d5a3d;
    --sage:     #4a7c5e;
    --mint:     #7fb896;
    --cream:    #f5f0e8;
    --warm:     #ede5d4;
    --clay:     #c4622d;
    --clay-lt:  #e8845a;
    --gold:     #d4a843;
    --text:     #1a2e22;
    --muted:    #5a7a65;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream) !important;
    color: var(--text) !important;
}

.stApp {
    background: linear-gradient(160deg, #f5f0e8 0%, #ede5d4 50%, #e8dfc8 100%) !important;
    min-height: 100vh;
}

/* ── Hide default streamlit stuff ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; max-width: 780px !important; }

/* ── Hero Header ── */
.hero {
    background: linear-gradient(135deg, var(--forest) 0%, var(--moss) 60%, var(--sage) 100%);
    border-radius: 24px;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(26,58,42,0.25);
}
.hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(127,184,150,0.2) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '🌍';
    position: absolute;
    bottom: 15px; right: 25px;
    font-size: 5rem;
    opacity: 0.15;
}
.hero h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 3rem !important;
    font-weight: 900 !important;
    color: var(--cream) !important;
    margin: 0 0 0.4rem !important;
    letter-spacing: -1px;
    line-height: 1.1;
}
.hero p {
    color: var(--mint) !important;
    font-size: 1.05rem !important;
    font-weight: 300 !important;
    margin: 0 !important;
    letter-spacing: 0.3px;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--forest);
    color: var(--cream) !important;
    padding: 0.7rem 1.2rem;
    border-radius: 12px;
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 1.8rem 0 1rem;
    letter-spacing: 0.2px;
}

/* ── Question Labels ── */
.stSelectbox label, .stSlider label, .stNumberInput label {
    font-weight: 500 !important;
    color: var(--forest) !important;
    font-size: 0.95rem !important;
    margin-bottom: 4px !important;
}

/* ── Selectbox styling ── */
.stSelectbox > div > div {
    background: white !important;
    border: 2px solid var(--warm) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    transition: border-color 0.2s;
}
.stSelectbox > div > div:hover {
    border-color: var(--sage) !important;
}

/* ── Number input ── */
.stNumberInput > div > div > input {
    background: white !important;
    border: 2px solid var(--warm) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── Submit Button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--clay) 0%, var(--clay-lt) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 2.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    margin-top: 1.5rem !important;
    cursor: pointer !important;
    box-shadow: 0 8px 25px rgba(196,98,45,0.35) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 35px rgba(196,98,45,0.45) !important;
}

/* ── Score Card ── */
.score-card {
    background: linear-gradient(135deg, var(--forest) 0%, var(--moss) 100%);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 2rem 0 1.5rem;
    box-shadow: 0 15px 45px rgba(26,58,42,0.2);
    position: relative;
    overflow: hidden;
}
.score-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 80% 20%, rgba(127,184,150,0.15) 0%, transparent 60%);
}
.score-number {
    font-family: 'Playfair Display', serif;
    font-size: 4.5rem;
    font-weight: 900;
    color: var(--gold);
    line-height: 1;
    margin-bottom: 0.2rem;
}
.score-label {
    font-size: 1.3rem;
    color: var(--cream);
    font-weight: 500;
    margin-bottom: 0.5rem;
}
.score-tier {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    color: var(--mint);
    padding: 0.3rem 1.2rem;
    border-radius: 20px;
    font-size: 0.95rem;
    font-weight: 500;
    border: 1px solid rgba(127,184,150,0.3);
}

/* ── Progress Bar (score gauge) ── */
.gauge-wrap {
    margin: 1rem 0 0.3rem;
}
.gauge-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 50px;
    height: 12px;
    overflow: hidden;
}
.gauge-bar-fill {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, var(--clay) 0%, var(--gold) 50%, var(--mint) 100%);
    transition: width 1s ease;
}
.gauge-labels {
    display: flex;
    justify-content: space-between;
    color: rgba(245,240,232,0.5);
    font-size: 0.75rem;
    margin-top: 4px;
}

/* ── Positive Card ── */
.positive-card {
    background: linear-gradient(135deg, #e8f5ec 0%, #d4edda 100%);
    border: 2px solid #a8d5b5;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
}
.positive-card h4 {
    color: var(--forest) !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1rem !important;
    margin: 0 0 0.7rem !important;
    font-weight: 700 !important;
}
.positive-item {
    color: var(--moss);
    font-size: 0.9rem;
    padding: 3px 0;
    font-weight: 400;
}

/* ── Recommendation Card ── */
.rec-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    border-left: 5px solid var(--clay);
    box-shadow: 0 4px 20px rgba(26,58,42,0.08);
    position: relative;
}
.rec-number {
    position: absolute;
    top: -10px; left: 16px;
    background: var(--clay);
    color: white;
    width: 24px; height: 24px;
    border-radius: 50%;
    font-size: 0.75rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
}
.rec-habit {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 1rem;
    padding-top: 0.5rem;
}
.rec-you {
    flex: 1;
    background: #fff0eb;
    border: 1.5px solid #f0c4b0;
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    font-size: 0.85rem;
    color: #7a3520;
}
.rec-you span { color: var(--clay); font-weight: 600; font-size: 0.75rem; display: block; margin-bottom: 2px; }
.rec-arrow { font-size: 1.3rem; padding-top: 0.6rem; color: var(--muted); }
.rec-them {
    flex: 1;
    background: #edf7f1;
    border: 1.5px solid #a8d5b5;
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    font-size: 0.85rem;
    color: #1a4d2e;
}
.rec-them span { color: var(--sage); font-weight: 600; font-size: 0.75rem; display: block; margin-bottom: 2px; }

.steps-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.5rem;
}
.step-easy   { background: #e8f5ec; border-radius: 8px; padding: 0.5rem 0.8rem; font-size: 0.85rem; margin-bottom: 5px; color: #2d6a4f; }
.step-medium { background: #fff8e1; border-radius: 8px; padding: 0.5rem 0.8rem; font-size: 0.85rem; margin-bottom: 5px; color: #856c00; }
.step-hard   { background: #fdecea; border-radius: 8px; padding: 0.5rem 0.8rem; font-size: 0.85rem; color: #b03a2e; }

/* ── Divider ── */
.eco-divider {
    text-align: center;
    color: var(--muted);
    font-size: 1.2rem;
    margin: 0.5rem 0;
    letter-spacing: 8px;
}

/* ── Footer ── */
.eco-footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 3rem;
    padding: 1rem;
    border-top: 1px solid var(--warm);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD & TRAIN MODEL (cached so it runs once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv("lifestyle_sustainability_data.csv")

    le_dict = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df.drop(['ParticipantID', 'Rating'], axis=1)
    Y = df['Rating']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.08,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1, reg_lambda=3, random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    return model, df, X, le_dict


# ─────────────────────────────────────────────
#  RECOMMENDATION DATA
# ─────────────────────────────────────────────
sustainability_rank = {
    'TransportationMode'  : {3: 4, 0: 3, 2: 2, 1: 1},
    'EnergySource'        : {2: 3, 0: 2, 1: 1},
    'UsingPlasticProducts': {1: 3, 2: 2, 0: 1},
    'DietType'            : {2: 3, 0: 2, 1: 1},
    'DisposalMethods'     : {1: 4, 3: 3, 0: 2, 2: 1},
    'ClothingFrequency'   : {1: 3, 2: 2, 0: 1},
    'SustainableBrands'   : {1: 2, 0: 1},
    'CommunityInvolvement': {0: 3, 2: 2, 1: 1},
    'LocalFoodFrequency'  : {0: 3, 2: 2, 1: 1},
}

feature_labels = {
    'TransportationMode'  : {0: 'Bicycle', 1: 'Car / Petrol vehicle', 2: 'Public transport', 3: 'Walking'},
    'EnergySource'        : {0: 'Mixed energy', 1: 'Regular grid electricity', 2: 'Solar / Renewable'},
    'UsingPlasticProducts': {0: 'Uses plastic often', 1: 'Rarely uses plastic', 2: 'Sometimes uses plastic'},
    'DietType'            : {0: 'Mixed diet', 1: 'Mostly non-veg', 2: 'Mostly veg / plant-based'},
    'DisposalMethods'     : {0: 'Mixed disposal', 1: 'Composting', 2: 'Single bin / landfill', 3: 'Separate recycling'},
    'ClothingFrequency'   : {0: 'Buys clothes often', 1: 'Rarely buys clothes', 2: 'Buys sometimes'},
    'SustainableBrands'   : {0: 'Does not buy eco brands', 1: 'Buys eco-friendly brands'},
    'CommunityInvolvement': {0: 'Very active', 1: 'Not involved', 2: 'Somewhat involved'},
    'LocalFoodFrequency'  : {0: 'Often eats local food', 1: 'Rarely eats local food', 2: 'Sometimes eats local'},
}

tiered_advice = {
    'TransportationMode': [
        "For short distances under 1km, try walking instead of taking the vehicle",
        "Use bus, auto, or metro for your regular commute at least 3 days a week",
        "Switch to cycling as your primary daily transport"
    ],
    'EnergySource': [
        "Switch off appliances fully instead of leaving on standby — saves 10-15% electricity",
        "Ask your electricity provider if they offer a green / solar energy plan",
        "Install a rooftop solar panel — many Indian states offer government subsidies"
    ],
    'UsingPlasticProducts': [
        "Carry one reusable cloth bag whenever you go to buy groceries",
        "Replace plastic water bottles with a steel or copper bottle",
        "Do a full audit of your kitchen and replace all single-use plastic containers"
    ],
    'DietType': [
        "Have one fully vegetarian day per week (like a Meatless Monday)",
        "Replace one non-veg meal per day with dal, rajma, or paneer",
        "Gradually shift to a mostly plant-based diet over 2-3 months"
    ],
    'DisposalMethods': [
        "Keep two separate dustbins — one for wet waste and one for dry waste",
        "Start composting your kitchen food scraps using a small compost bin",
        "Set up full waste segregation at home — wet, dry, and hazardous separately"
    ],
    'ClothingFrequency': [
        "Before buying new clothes, ask yourself if you really need them",
        "Try buying from thrift stores or second-hand apps like OLX for clothes",
        "Commit to buying no new clothing for 3 months — use what you have"
    ],
    'SustainableBrands': [
        "Next time you buy a product, check if there is an eco-friendly version",
        "Replace 2-3 regular household products with eco-certified alternatives",
        "Make sustainable brands your default choice for all future shopping"
    ],
    'CommunityInvolvement': [
        "Share one environmental tip or article on your social media this week",
        "Join a local cleanliness drive or tree plantation event in your area",
        "Regularly volunteer with an environmental NGO or start an eco-initiative in your colony"
    ],
    'LocalFoodFrequency': [
        "Visit your nearest local sabzi mandi or farmer market once this week",
        "Buy fruits and vegetables from local vendors instead of supermarkets",
        "Source at least 70% of your food from local markets regularly"
    ],
}

positive_acknowledgements = {
    'TransportationMode'  : {3: "You walk for most trips 🚶", 0: "You cycle regularly 🚲", 2: "You use public transport 🚌"},
    'EnergySource'        : {2: "You use solar / renewable energy ⚡🌞"},
    'UsingPlasticProducts': {1: "You rarely use plastic 🧴✅"},
    'DietType'            : {2: "You follow a mostly plant-based diet 🥗"},
    'DisposalMethods'     : {1: "You compost your waste ♻️", 3: "You already recycle ♻️"},
    'ClothingFrequency'   : {1: "You rarely buy new clothes 👗✅"},
    'SustainableBrands'   : {1: "You buy from sustainable brands 🛒✅"},
    'CommunityInvolvement': {0: "You're very active in your community 🤝"},
    'LocalFoodFrequency'  : {0: "You often eat locally sourced food 🌾"},
}


# ─────────────────────────────────────────────
#  RECOMMENDATION FUNCTION
# ─────────────────────────────────────────────
def get_recommendations(user_input_dict, user_score, df_original, X):
    user_array = np.array(
        [user_input_dict.get(col, 0) for col in X.columns]
    ).reshape(1, -1)

    similarities = cosine_similarity(user_array, X.values)[0]
    similar_indices = np.argsort(similarities)[::-1][:20]

    better_users_indices = [
        idx for idx in similar_indices
        if df_original.iloc[idx]['Rating'] > user_score
    ]
    if len(better_users_indices) == 0:
        better_users_indices = list(similar_indices[:10])

    actionable_features = list(sustainability_rank.keys())
    feature_improvement_count = {}
    feature_best_value = {}

    for feature in actionable_features:
        user_val  = user_input_dict.get(feature, 0)
        user_rank = sustainability_rank[feature].get(int(user_val), 0)
        better_count = 0
        value_counts = {}
        for idx in better_users_indices:
            their_val  = int(X.iloc[idx][feature])
            their_rank = sustainability_rank[feature].get(their_val, 0)
            if their_rank > user_rank:
                better_count += 1
                value_counts[their_val] = value_counts.get(their_val, 0) + 1
        if better_count > 0:
            feature_improvement_count[feature] = better_count
            feature_best_value[feature] = max(value_counts, key=value_counts.get)

    sorted_improvements = sorted(
        feature_improvement_count.items(), key=lambda x: x[1], reverse=True
    )

    positives = []
    for feature in actionable_features:
        user_val = int(user_input_dict.get(feature, 0))
        ack = positive_acknowledgements.get(feature, {}).get(user_val)
        if ack:
            positives.append(ack)

    recommendations = []
    for feature, count in sorted_improvements[:4]:
        user_val     = int(user_input_dict.get(feature, 0))
        best_val     = feature_best_value[feature]
        user_label   = feature_labels[feature].get(user_val,  str(user_val))
        better_label = feature_labels[feature].get(best_val, str(best_val))
        steps        = tiered_advice[feature]
        recommendations.append({
            'user_label'  : user_label,
            'better_label': better_label,
            'steps'       : steps,
        })

    return positives, recommendations


# ─────────────────────────────────────────────
#  SCORE TIER
# ─────────────────────────────────────────────
def get_tier(score):
    if score >= 4.5: return "🏆 Sustainability Champion"
    if score >= 3.5: return "🌿 Advanced"
    if score >= 2.5: return "🌱 Intermediate"
    return "⚠️ Needs Improvement"


# ─────────────────────────────────────────────
#  UI — HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>SustainMe ♻️</h1>
    <p>Discover your sustainability score and get personalised advice based on people like you.</p>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("Loading model..."):
    model, df, X, le_dict = load_model()


# ─────────────────────────────────────────────
#  FORM
# ─────────────────────────────────────────────
with st.form("sustainability_form"):

    # ── About You ──
    st.markdown('<div class="section-header">👤 About You</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Your age", min_value=10, max_value=100, value=20)
    with col2:
        gender = st.selectbox("Gender", ["Female", "Male", "Other / Prefer not to say"])
    location = st.selectbox("Where do you live?", [
        "Village / Small town",
        "Outskirts of a city (like Gandhinagar near Ahmedabad)",
        "City center (like Ahmedabad, Mumbai, Delhi)"
    ])

    # ── Diet & Food ──
    st.markdown('<div class="section-header">🥗 Diet & Food</div>', unsafe_allow_html=True)
    diet = st.selectbox("What best describes your daily diet?", [
        "Mix of everything (dal, roti, chicken, veggies etc.)",
        "Mostly non-veg (chicken, fish, eggs, meat)",
        "Mostly veg / vegan (vegetables, fruits, lentils)"
    ])
    local_food = st.selectbox("How often do you eat locally grown food?", [
        "Often (most of the time)",
        "Rarely (almost never)",
        "Sometimes (occasionally)"
    ])

    # ── Transport ──
    st.markdown('<div class="section-header">🚗 Transportation</div>', unsafe_allow_html=True)
    transport = st.selectbox("What is your main way of getting around daily?", [
        "Bicycle",
        "Car / bike (petrol/diesel vehicle)",
        "Bus / auto / metro / rickshaw",
        "I mostly walk"
    ])

    # ── Energy & Home ──
    st.markdown('<div class="section-header">⚡ Energy & Home</div>', unsafe_allow_html=True)
    energy = st.selectbox("What type of electricity do you use at home?", [
        "Mix (sometimes solar, sometimes regular grid)",
        "Regular electricity from the grid (coal/gas based)",
        "Solar panels or green energy provider"
    ])
    col3, col4 = st.columns(2)
    with col3:
        electricity = st.number_input(
            "Monthly electricity (units/kWh from bill)",
            min_value=0, max_value=2000, value=250,
            help="Check your electricity bill — look for 'Units' or 'kWh'. Typical: small flat=100-200, medium house=200-400"
        )
    with col4:
        home_type = st.selectbox("Home type", ["Flat / Apartment", "Independent house / Bungalow"])

    home_size_option = st.selectbox("How big is your home?", [
        "Small (1BHK or studio)",
        "Medium (2BHK)",
        "Large (3BHK or bigger)"
    ])

    # ── Shopping & Waste ──
    st.markdown('<div class="section-header">🛍️ Shopping & Waste</div>', unsafe_allow_html=True)
    plastic = st.selectbox("How often do you use plastic bags, bottles, or packaging?", [
        "Often (daily use of plastic bags, plastic bottles etc.)",
        "Rarely (I carry reusable bags, avoid plastic)",
        "Sometimes (I try to avoid but not always)"
    ])
    clothing = st.selectbox("How often do you buy new clothes?", [
        "Often (every month or more)",
        "Rarely (only when really needed)",
        "Sometimes (a few times a year)"
    ])
    sustainable_brands = st.selectbox("Do you buy from eco-friendly or sustainable brands?", [
        "No",
        "Yes"
    ])
    disposal = st.selectbox("How do you usually throw away your waste / garbage?", [
        "Mix of different methods",
        "Composting (wet waste like food scraps go to compost)",
        "Everything goes in one bin (landfill / regular garbage)",
        "I separate and recycle (paper, plastic, glass separately)"
    ])

    # ── Awareness & Lifestyle ──
    st.markdown('<div class="section-header">🌍 Awareness & Lifestyle</div>', unsafe_allow_html=True)
    awareness = st.selectbox("How aware are you about environmental issues?", [
        "1 — Not aware at all",
        "2 — Slightly aware",
        "3 — Moderately aware",
        "4 — Quite aware",
        "5 — Very aware, I actively follow this topic"
    ])
    community = st.selectbox("Are you involved in any community or social activities?", [
        "Yes, very active (colony events, NGOs, cleanliness drives)",
        "No, not really",
        "Sometimes / a little"
    ])
    physical = st.selectbox("How physically active are you in daily life?", [
        "Very active (daily exercise or physical work)",
        "Not very active (mostly sitting)",
        "Moderately active (some days active, some not)"
    ])
    water_option = st.selectbox("How much water does your household use per month?", [
        "Less (under 2000 liters — small flat, 1-2 people)",
        "Medium (2000-5000 liters — average family)",
        "High (above 5000 liters — large house or family)"
    ])

    submitted = st.form_submit_button("🌱 Check My Sustainability Score!")


# ─────────────────────────────────────────────
#  PROCESS & SHOW RESULTS
# ─────────────────────────────────────────────
if submitted:

    # ── Map friendly answers back to encoded numbers ──
    gender_map    = {"Female": 0, "Male": 1, "Other / Prefer not to say": 2}
    location_map  = {
        "Village / Small town": 0,
        "Outskirts of a city (like Gandhinagar near Ahmedabad)": 1,
        "City center (like Ahmedabad, Mumbai, Delhi)": 2
    }
    diet_map      = {
        "Mix of everything (dal, roti, chicken, veggies etc.)": 0,
        "Mostly non-veg (chicken, fish, eggs, meat)": 1,
        "Mostly veg / vegan (vegetables, fruits, lentils)": 2
    }
    local_food_map = {
        "Often (most of the time)": 0,
        "Rarely (almost never)": 1,
        "Sometimes (occasionally)": 2
    }
    transport_map = {
        "Bicycle": 0,
        "Car / bike (petrol/diesel vehicle)": 1,
        "Bus / auto / metro / rickshaw": 2,
        "I mostly walk": 3
    }
    energy_map    = {
        "Mix (sometimes solar, sometimes regular grid)": 0,
        "Regular electricity from the grid (coal/gas based)": 1,
        "Solar panels or green energy provider": 2
    }
    home_type_map = {"Flat / Apartment": 0, "Independent house / Bungalow": 1}
    home_size_map = {"Small (1BHK or studio)": 500, "Medium (2BHK)": 900, "Large (3BHK or bigger)": 1600}
    plastic_map   = {
        "Often (daily use of plastic bags, plastic bottles etc.)": 0,
        "Rarely (I carry reusable bags, avoid plastic)": 1,
        "Sometimes (I try to avoid but not always)": 2
    }
    clothing_map  = {
        "Often (every month or more)": 0,
        "Rarely (only when really needed)": 1,
        "Sometimes (a few times a year)": 2
    }
    brands_map    = {"No": 0, "Yes": 1}
    disposal_map  = {
        "Mix of different methods": 0,
        "Composting (wet waste like food scraps go to compost)": 1,
        "Everything goes in one bin (landfill / regular garbage)": 2,
        "I separate and recycle (paper, plastic, glass separately)": 3
    }
    awareness_map = {
        "1 — Not aware at all": 1, "2 — Slightly aware": 2,
        "3 — Moderately aware": 3, "4 — Quite aware": 4,
        "5 — Very aware, I actively follow this topic": 5
    }
    community_map = {
        "Yes, very active (colony events, NGOs, cleanliness drives)": 0,
        "No, not really": 1,
        "Sometimes / a little": 2
    }
    physical_map  = {
        "Very active (daily exercise or physical work)": 0,
        "Not very active (mostly sitting)": 1,
        "Moderately active (some days active, some not)": 2
    }
    water_map     = {
        "Less (under 2000 liters — small flat, 1-2 people)": 1500,
        "Medium (2000-5000 liters — average family)": 3500,
        "High (above 5000 liters — large house or family)": 6000
    }

    # Build user dict
    full_user = {
        'Age'                          : age,
        'Gender'                       : gender_map[gender],
        'Location'                     : location_map[location],
        'DietType'                     : diet_map[diet],
        'LocalFoodFrequency'           : local_food_map[local_food],
        'TransportationMode'           : transport_map[transport],
        'EnergySource'                 : energy_map[energy],
        'MonthlyElectricityConsumption': electricity,
        'HomeType'                     : home_type_map[home_type],
        'HomeSize'                     : home_size_map[home_size_option],
        'UsingPlasticProducts'         : plastic_map[plastic],
        'ClothingFrequency'            : clothing_map[clothing],
        'SustainableBrands'            : brands_map[sustainable_brands],
        'DisposalMethods'              : disposal_map[disposal],
        'EnvironmentalAwareness'       : awareness_map[awareness],
        'CommunityInvolvement'         : community_map[community],
        'PhysicalActivities'           : physical_map[physical],
        'MonthlyWaterConsumption'      : water_map[water_option],
    }

    # Build input array
    input_data = [full_user.get(col, 0) for col in X.columns]

    # Predict
    score = float(model.predict([input_data])[0])
    score_display = round(score, 2)
    tier = get_tier(score_display)
    gauge_pct = int((score / 5) * 100)

    # ── Score Card ──
    st.markdown(f"""
    <div class="score-card">
        <div class="score-number">{score_display}</div>
        <div class="score-label">out of 5.0</div>
        <div class="score-tier">{tier}</div>
        <div class="gauge-wrap">
            <div class="gauge-bar-bg">
                <div class="gauge-bar-fill" style="width:{gauge_pct}%"></div>
            </div>
            <div class="gauge-labels"><span>1 — Low</span><span>3 — Medium</span><span>5 — High</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Get recommendations
    positives, recommendations = get_recommendations(full_user, score, df, X)

    # ── What you're doing well ──
    if positives:
        items_html = "".join([f'<div class="positive-item">✅ {p}</div>' for p in positives])
        st.markdown(f"""
        <div class="positive-card">
            <h4>🌟 What you're already doing great</h4>
            {items_html}
        </div>
        """, unsafe_allow_html=True)

    # ── Recommendations ──
    if recommendations:
        st.markdown('<div class="section-header">📌 Areas to Improve</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#5a7a65; font-size:0.9rem; margin: -0.5rem 0 1rem;">Based on people similar to you who scored higher</p>',
            unsafe_allow_html=True
        )
        for i, rec in enumerate(recommendations, 1):
            steps = rec['steps']
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-number">{i}</div>
                <div class="rec-habit">
                    <div class="rec-you">
                        <span>YOU</span>
                        {rec['user_label']}
                    </div>
                    <div class="rec-arrow">→</div>
                    <div class="rec-them">
                        <span>BETTER HABIT</span>
                        {rec['better_label']}
                    </div>
                </div>
                <div class="steps-title">How to improve — pick your level</div>
                <div class="step-easy">🟢 Easy &nbsp;&nbsp; {steps[0]}</div>
                <div class="step-medium">🟡 Medium &nbsp; {steps[1]}</div>
                <div class="step-hard">🔴 Hard &nbsp;&nbsp; {steps[2]}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="positive-card">
            <h4>🏆 You're already near the top!</h4>
            <div class="positive-item">Your lifestyle is very close to the highest scorers in our dataset. Consider inspiring others in your community!</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="eco-divider">🌿 · · · 🌿</div>', unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div class="eco-footer">
    SustainMe ♻️ — Built with 💚 using Streamlit & XGBoost
</div>
""", unsafe_allow_html=True)