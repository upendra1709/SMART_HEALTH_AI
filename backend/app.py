"""
Smart Health AI — Flask Backend for Dataset
=======================================================
Works with the model trained by train_model.py

Covers all 42 diseases in the dataset:
  Fungal infection, Allergy, GERD, Chronic cholestasis,
  Drug Reaction, Peptic ulcer disease, AIDS, Diabetes,
  Gastroenteritis, Bronchial Asthma, Hypertension,
  Migraine, Cervical spondylosis, Paralysis (brain hemorrhage),
  Jaundice, Malaria, Chicken pox, Dengue, Typhoid,
  hepatitis A/B/C/D/E, Alcoholic hepatitis, Tuberculosis,
  Common Cold, Pneumonia, Dimorphic hemmorhoids(piles),
  Heart attack, Varicose veins, Hypothyroidism, Hyperthyroidism,
  Hypoglycemia, Osteoarthritis, Arthritis,
  (vertigo) Paroxysmal Positional Vertigo, Acne,
  Urinary tract infection, Psoriasis, Impetigo
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, os
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# ── Load Model ────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

model           = joblib.load(os.path.join(MODEL_DIR, "disease_model.pkl"))
label_encoder   = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
symptom_columns = joblib.load(os.path.join(MODEL_DIR, "model_features.pkl"))

print(f"✅ Model loaded   : {len(model.classes_)} diseases")
print(f"✅ Symptoms loaded: {len(symptom_columns)} symptoms")

# ── Symptom Matching ──────────────────────────────────────
# Handles: "Skin Rash", "skin rash", "skin_rash" → "skin_rash"
def _norm(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_")

SYMPTOM_MAP: dict[str, str] = {}
for col in symptom_columns:
    SYMPTOM_MAP[_norm(col)] = col
    SYMPTOM_MAP[col.replace("_", " ")] = col   # space variant

def find_symptom(raw: str):
    key = _norm(raw)
    if key in SYMPTOM_MAP:
        return SYMPTOM_MAP[key]
    # substring fallback
    for col in symptom_columns:
        if key in _norm(col) or _norm(col) in key:
            return col
    return None

# ── Disease Info (all 42 diseases) ───────────────────────
# Keys match EXACTLY what label_encoder stores (stripped, original case)
DISEASE_INFO = {
    "Fungal infection": {
        "description": "A fungal infection occurs when fungi invade and grow in the body, commonly affecting skin, nails, or mucous membranes.",
        "precautions": ["Keep affected areas clean and dry", "Use prescribed antifungal medication", "Avoid sharing personal items", "Wear breathable clothing"],
        "risk": "low"
    },
    "Allergy": {
        "description": "An allergy is an immune system overreaction to a normally harmless substance such as pollen, pet dander, or certain foods.",
        "precautions": ["Identify and avoid allergens", "Take prescribed antihistamines", "Keep emergency medication (EpiPen) nearby if severe", "Consult an allergist"],
        "risk": "low"
    },
    "GERD": {
        "description": "Gastroesophageal Reflux Disease is a chronic condition where stomach acid repeatedly flows back into the esophagus, causing irritation.",
        "precautions": ["Avoid spicy, fatty, and acidic foods", "Eat smaller meals and don't lie down after eating", "Elevate head while sleeping", "Consult a gastroenterologist"],
        "risk": "medium"
    },
    "Chronic cholestasis": {
        "description": "Chronic cholestasis is reduced or blocked bile flow from the liver, causing bile buildup and liver damage over time.",
        "precautions": ["Avoid alcohol and fatty foods", "Take prescribed medications regularly", "Get regular liver function tests", "Consult a hepatologist"],
        "risk": "high"
    },
    "Drug Reaction": {
        "description": "A drug reaction is an adverse response to a medication, ranging from mild skin rashes to severe life-threatening reactions.",
        "precautions": ["Stop the suspected medication immediately", "Seek emergency medical attention", "Document the reaction clearly", "Inform all healthcare providers about it"],
        "risk": "high"
    },
    "Peptic ulcer disease": {
        "description": "Peptic ulcers are painful sores on the stomach lining or small intestine, usually caused by H. pylori bacteria or long-term NSAID use.",
        "precautions": ["Avoid NSAIDs, alcohol, and smoking", "Take prescribed antibiotics or PPIs fully", "Eat small, regular meals", "Manage stress effectively"],
        "risk": "medium"
    },
    "AIDS": {
        "description": "AIDS is the advanced stage of HIV infection, severely damaging the immune system and making the body vulnerable to infections and cancers.",
        "precautions": ["Take antiretroviral therapy (ART) consistently", "Use protection during sexual activity", "Avoid sharing needles", "Get regular CD4 count and viral load tests"],
        "risk": "high"
    },
    "Diabetes": {
        "description": "Diabetes is a chronic metabolic disease where the body cannot properly regulate blood sugar, either due to insufficient insulin or insulin resistance.",
        "precautions": ["Monitor blood sugar levels daily", "Follow a low-sugar, high-fiber diet", "Exercise at least 30 minutes daily", "Take prescribed medications consistently"],
        "risk": "high"
    },
    "Gastroenteritis": {
        "description": "Gastroenteritis is inflammation of the stomach and intestines from bacterial or viral infection, causing vomiting, diarrhea, and stomach pain.",
        "precautions": ["Stay well hydrated with oral rehydration salts", "Rest and eat bland foods (BRAT diet)", "Wash hands thoroughly and frequently", "See a doctor if symptoms last more than 2 days"],
        "risk": "medium"
    },
    "Bronchial Asthma": {
        "description": "Bronchial asthma is a chronic respiratory condition where airway inflammation and narrowing cause episodes of wheezing, breathlessness, and chest tightness.",
        "precautions": ["Always carry prescribed rescue inhaler", "Identify and avoid personal triggers", "Monitor peak flow regularly", "Seek emergency care during severe attacks"],
        "risk": "medium"
    },
    "Hypertension ": {
        "description": "Hypertension (high blood pressure) is a chronic condition where blood pushes too hard against artery walls, increasing risk of heart disease and stroke.",
        "precautions": ["Reduce salt and processed food intake", "Exercise regularly and maintain healthy weight", "Manage stress with relaxation techniques", "Take blood pressure medications as prescribed"],
        "risk": "high"
    },
    "Migraine": {
        "description": "Migraine is a neurological condition causing intense, throbbing headaches often accompanied by nausea, vomiting, and sensitivity to light and sound.",
        "precautions": ["Identify and avoid personal triggers (caffeine, stress, certain foods)", "Rest in a dark, quiet room during attacks", "Take prescribed migraine medication at onset", "Keep a headache diary to track patterns"],
        "risk": "medium"
    },
    "Cervical spondylosis": {
        "description": "Cervical spondylosis is age-related degeneration of the neck's spinal disks and joints, causing neck pain, stiffness, and sometimes nerve compression.",
        "precautions": ["Maintain good posture at all times", "Use an ergonomic chair and monitor setup", "Do physiotherapist-prescribed neck exercises", "Apply heat/ice packs for pain relief"],
        "risk": "medium"
    },
    "Paralysis (brain hemorrhage)": {
        "description": "Brain hemorrhage is bleeding inside or around the brain from a ruptured blood vessel, which can cause sudden paralysis and is life-threatening.",
        "precautions": ["EMERGENCY — call ambulance immediately (112/108)", "Do not move the patient unnecessarily", "Keep airways clear and monitor breathing", "Urgent neurosurgical evaluation needed"],
        "risk": "high"
    },
    "Jaundice": {
        "description": "Jaundice is a yellowing of the skin and whites of the eyes caused by excess bilirubin, indicating liver, bile duct, or blood breakdown problems.",
        "precautions": ["Avoid alcohol completely", "Stay well hydrated", "Follow doctor's prescribed treatment plan", "Get liver function tests and ultrasound done"],
        "risk": "high"
    },
    "Malaria": {
        "description": "Malaria is a life-threatening disease caused by Plasmodium parasites transmitted through the bite of infected female Anopheles mosquitoes.",
        "precautions": ["Take prescribed antimalarial medication immediately", "Use mosquito repellent and bed nets", "Wear long-sleeved clothing in affected areas", "Seek medical attention at first fever signs"],
        "risk": "high"
    },
    "Chicken pox": {
        "description": "Chickenpox is a highly contagious viral disease caused by the varicella-zoster virus, producing an itchy blister rash across the body.",
        "precautions": ["Isolate to prevent spreading to others", "Apply calamine lotion to reduce itching", "Keep fingernails short and avoid scratching", "Rest and stay well hydrated"],
        "risk": "low"
    },
    "Dengue": {
        "description": "Dengue fever is a painful mosquito-borne viral infection that can progress to severe dengue, causing internal bleeding and organ damage.",
        "precautions": ["Seek medical attention immediately — platelet monitoring is critical", "Use mosquito repellent and eliminate standing water", "Stay well hydrated", "Avoid aspirin and ibuprofen"],
        "risk": "high"
    },
    "Typhoid": {
        "description": "Typhoid fever is a bacterial infection caused by Salmonella typhi, spread through contaminated food and water, causing prolonged high fever.",
        "precautions": ["Complete full antibiotic course as prescribed", "Drink only boiled or purified water", "Practice strict food and hand hygiene", "Get the typhoid vaccine for future prevention"],
        "risk": "high"
    },
    "hepatitis A": {
        "description": "Hepatitis A is a highly contagious liver infection caused by the hepatitis A virus, spread through contaminated food or water.",
        "precautions": ["Rest adequately and avoid alcohol", "Stay well hydrated", "Practice strict hand hygiene after toilet use", "Get vaccinated to prevent future infection"],
        "risk": "medium"
    },
    "Hepatitis B": {
        "description": "Hepatitis B is a serious liver infection caused by HBV that can become chronic and lead to cirrhosis, liver failure, or liver cancer.",
        "precautions": ["Avoid alcohol completely", "Take antiviral medications as prescribed", "Get vaccinated (highly effective prevention)", "Monitor liver health with regular tests"],
        "risk": "high"
    },
    "Hepatitis C": {
        "description": "Hepatitis C is a bloodborne viral infection causing liver inflammation that often becomes chronic without symptoms until serious liver damage occurs.",
        "precautions": ["Never share needles, syringes, or personal items", "Avoid alcohol which accelerates liver damage", "Take direct-acting antiviral treatment (95% cure rate)", "Get regular liver fibrosis assessments"],
        "risk": "high"
    },
    "Hepatitis D": {
        "description": "Hepatitis D only occurs in people already infected with hepatitis B. Co-infection leads to more severe liver disease than either virus alone.",
        "precautions": ["Treat underlying Hepatitis B first", "Avoid alcohol completely", "Get regular liver function tests", "Consult a hepatologist urgently"],
        "risk": "high"
    },
    "Hepatitis E": {
        "description": "Hepatitis E is a waterborne liver infection especially dangerous during pregnancy, prevalent in areas with poor water sanitation.",
        "precautions": ["Drink only boiled or bottled water", "Maintain strict food and hand hygiene", "Rest and stay hydrated", "Seek urgent medical care if pregnant"],
        "risk": "high"
    },
    "Alcoholic hepatitis": {
        "description": "Alcoholic hepatitis is liver inflammation caused by heavy, prolonged alcohol consumption — a serious and potentially fatal condition.",
        "precautions": ["Stop all alcohol consumption immediately and permanently", "Follow a nutritious, high-calorie diet", "Take prescribed corticosteroids if advised", "Attend alcohol dependency counselling"],
        "risk": "high"
    },
    "Tuberculosis": {
        "description": "Tuberculosis (TB) is a highly contagious bacterial infection caused by Mycobacterium tuberculosis that primarily attacks the lungs.",
        "precautions": ["Complete the full 6–9 month antibiotic course without missing doses", "Wear a mask and isolate during the infectious period", "Cover mouth when coughing or sneezing", "Get all household contacts tested"],
        "risk": "high"
    },
    "Common Cold": {
        "description": "The common cold is a mild viral upper respiratory infection affecting the nose and throat, usually resolving on its own within 7–10 days.",
        "precautions": ["Rest and stay well hydrated", "Take over-the-counter cold remedies for symptom relief", "Wash hands frequently to avoid spreading", "Avoid close contact with vulnerable people"],
        "risk": "low"
    },
    "Pneumonia": {
        "description": "Pneumonia is a serious lung infection causing air sacs to fill with fluid or pus, making breathing difficult and potentially life-threatening.",
        "precautions": ["Complete the full course of prescribed antibiotics", "Rest and drink plenty of fluids", "Monitor oxygen levels — seek ER if below 94%", "Get pneumococcal vaccine for future prevention"],
        "risk": "high"
    },
    "Dimorphic hemmorhoids(piles)": {
        "description": "Hemorrhoids (piles) are swollen, inflamed veins in the rectum or anus causing pain, itching, and rectal bleeding.",
        "precautions": ["Eat a high-fiber diet and drink plenty of water", "Avoid straining during bowel movements", "Take sitz baths for pain relief", "Consult a proctologist if bleeding persists"],
        "risk": "medium"
    },
    "Heart attack": {
        "description": "A heart attack (myocardial infarction) occurs when blood supply to part of the heart is blocked, causing heart muscle cells to die.",
        "precautions": ["EMERGENCY — call 112/108 immediately", "Chew one regular aspirin (325mg) if not allergic", "Rest completely — do not walk or exert yourself", "Start CPR if the person becomes unresponsive"],
        "risk": "high"
    },
    "Varicose veins": {
        "description": "Varicose veins are enlarged, twisted veins visible beneath the skin surface, usually in the legs, caused by weakened or damaged vein valves.",
        "precautions": ["Exercise regularly, especially walking and swimming", "Elevate your legs above heart level when resting", "Wear prescribed compression stockings daily", "Consult a vascular surgeon for advanced cases"],
        "risk": "low"
    },
    "Hypothyroidism": {
        "description": "Hypothyroidism is an underactive thyroid gland that doesn't produce enough thyroid hormone, slowing down the body's metabolism.",
        "precautions": ["Take levothyroxine (thyroid hormone replacement) daily at same time", "Get thyroid function tests every 6–12 months", "Eat a balanced diet including selenium and iodine", "Report symptoms of under or over-treatment to doctor"],
        "risk": "medium"
    },
    "Hyperthyroidism": {
        "description": "Hyperthyroidism is an overactive thyroid gland producing excess thyroid hormone, speeding up the body's metabolism and causing various symptoms.",
        "precautions": ["Take prescribed anti-thyroid medications (methimazole/carbimazole)", "Avoid iodine-rich foods and iodine supplements", "Monitor heart rate regularly", "Discuss radioiodine or surgery options with endocrinologist"],
        "risk": "medium"
    },
    "Hypoglycemia": {
        "description": "Hypoglycemia is abnormally low blood sugar that deprives the body's cells of glucose energy, causing confusion, shakiness, and loss of consciousness.",
        "precautions": ["Always carry fast-acting glucose (glucose tablets, juice, candy)", "Eat regular meals — never skip meals", "Monitor blood glucose levels frequently", "Review diabetes medications with your doctor"],
        "risk": "medium"
    },
    "Osteoarthritis": {
        "description": "Osteoarthritis is the most common form of arthritis — a degenerative joint disease where protective cartilage wears down over time.",
        "precautions": ["Exercise regularly with low-impact activities (swimming, walking)", "Maintain a healthy weight to reduce joint stress", "Use joint support braces when needed", "Consult an orthopedic specialist about treatment options"],
        "risk": "medium"
    },
    "Arthritis": {
        "description": "Arthritis is inflammation of one or more joints causing pain, swelling, and stiffness that can worsen progressively with age.",
        "precautions": ["Exercise regularly but avoid high-impact activities", "Apply heat/cold therapy for pain relief", "Take prescribed anti-inflammatory medications", "Consult a rheumatologist for a long-term management plan"],
        "risk": "medium"
    },
    "(vertigo) Paroxysmal  Positional Vertigo": {
        "description": "Benign Paroxysmal Positional Vertigo (BPPV) is brief, intense episodes of dizziness triggered by specific head movements, caused by displaced inner ear crystals.",
        "precautions": ["Perform Epley maneuver exercises as taught by physiotherapist", "Move head slowly and deliberately", "Sit or lie down immediately when dizziness occurs", "Consult an ENT specialist or neurologist"],
        "risk": "low"
    },
    "Acne": {
        "description": "Acne is a skin condition where hair follicles become clogged with oil and dead skin cells, leading to pimples, blackheads, and whiteheads.",
        "precautions": ["Wash face twice daily with a gentle, non-comedogenic cleanser", "Avoid touching or picking at your face", "Use oil-free, non-comedogenic skincare products", "Consult a dermatologist for persistent or severe acne"],
        "risk": "low"
    },
    "Urinary tract infection": {
        "description": "A UTI is a bacterial infection in any part of the urinary system — kidneys, ureters, bladder, or urethra — most commonly the bladder.",
        "precautions": ["Drink at least 8 glasses of water daily", "Complete the full antibiotic course even if symptoms improve", "Wipe front to back after toilet use", "Urinate soon after sexual activity"],
        "risk": "medium"
    },
    "Psoriasis": {
        "description": "Psoriasis is a chronic autoimmune skin condition that accelerates the skin cell lifecycle, forming red, scaly, itchy patches on the skin surface.",
        "precautions": ["Moisturise skin daily with thick, fragrance-free creams", "Identify and avoid triggers (stress, alcohol, certain medications)", "Use prescribed topical corticosteroids or biologics", "Protect skin from injuries and infections"],
        "risk": "low"
    },
    "Impetigo": {
        "description": "Impetigo is a highly contagious bacterial skin infection common in children, causing red sores that rupture and form honey-colored crusts.",
        "precautions": ["Apply prescribed antibiotic ointment (mupirocin) to sores", "Keep sores clean and covered with loose bandages", "Wash hands frequently and don't touch sores", "Avoid school/work until sores are healed or 48hrs of antibiotics done"],
        "risk": "low"
    },
}

def get_disease_info(disease_name: str) -> dict:
    """Case-insensitive lookup with partial match fallback."""
    # Try exact match first
    if disease_name in DISEASE_INFO:
        return DISEASE_INFO[disease_name]
    # Try case-insensitive
    key_lower = disease_name.strip().lower()
    for k, v in DISEASE_INFO.items():
        if k.strip().lower() == key_lower:
            return v
    # Partial match
    for k, v in DISEASE_INFO.items():
        if key_lower in k.strip().lower() or k.strip().lower() in key_lower:
            return v
    # Fallback
    return {
        "description": f"{disease_name} requires professional medical evaluation. Please consult a qualified doctor.",
        "precautions": ["Consult a doctor immediately", "Rest and stay hydrated", "Monitor your symptoms closely", "Avoid self-medication"],
        "risk": "medium"
    }

# ── Routes ────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Smart Health AI — kaushil268 Dataset Backend",
        "version": "4.0.0",
        "diseases": len(model.classes_),
        "symptoms": len(symptom_columns),
        "endpoints": {
            "/predict":  "POST — predict disease from symptoms",
            "/symptoms": "GET  — list all valid symptoms",
            "/diseases": "GET  — list all 42 diseases"
        }
    })


@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    """Return all 132 symptoms in both raw and formatted form."""
    formatted = [s.replace("_", " ").title() for s in symptom_columns]
    return jsonify({
        "symptoms":  symptom_columns,
        "formatted": formatted,
        "count":     len(symptom_columns)
    })


@app.route("/diseases", methods=["GET"])
def get_diseases():
    """Return all 42 disease names the model can predict."""
    return jsonify({
        "diseases": sorted(label_encoder.classes_.tolist()),
        "count":    len(label_encoder.classes_)
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict disease from a list of symptoms.

    Request body:
        { "symptoms": ["itching", "skin_rash", "fever"] }

    Response:
        {
            "disease":            str,
            "confidence":         float,
            "description":        str,
            "precautions":        list[str],
            "risk":               str,           — "low" / "medium" / "high"
            "symptoms_matched":   list[str],      — recognised symptoms used
            "symptoms_not_found": list[str],      — inputs that didn't match
            "top_predictions":    list[{disease, confidence}]
        }
    """
    try:
        data = request.get_json()
        if not data or "symptoms" not in data:
            return jsonify({"error": "Missing 'symptoms' key in request body"}), 400

        user_symptoms = data["symptoms"]
        if not user_symptoms:
            return jsonify({"error": "Provide at least one symptom"}), 400

        # Build the 132-element binary feature vector
        sample             = dict.fromkeys(symptom_columns, 0)
        matched            = []
        not_found          = []

        for s in user_symptoms:
            canonical = find_symptom(str(s))
            if canonical:
                sample[canonical] = 1
                matched.append(canonical)
            else:
                not_found.append(s)

        if not matched:
            return jsonify({
                "error": "None of the provided symptoms were recognised. "
                         "Call GET /symptoms to see all valid symptom names.",
                "symptoms_not_found": not_found
            }), 400

        input_df      = pd.DataFrame([sample], columns=symptom_columns).astype("int8")
        probabilities = model.predict_proba(input_df)[0]

        top_idx           = int(np.argmax(probabilities))
        confidence        = float(probabilities[top_idx]) * 100
        predicted_disease = label_encoder.inverse_transform([top_idx])[0]

        # Top 5 predictions
        top5 = np.argsort(probabilities)[::-1][:5]
        top_predictions = [
            {
                "disease":    label_encoder.inverse_transform([int(i)])[0],
                "confidence": round(float(probabilities[i]) * 100, 2)
            }
            for i in top5 if probabilities[i] > 0.005
        ]

        info = get_disease_info(predicted_disease)

        return jsonify({
            "disease":             predicted_disease,
            "confidence":          round(confidence, 2),
            "description":         info["description"],
            "precautions":         info["precautions"],
            "risk":                info["risk"],
            "symptoms_matched":    matched,
            "symptoms_not_found":  not_found,
            "top_predictions":     top_predictions
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n🏥  Smart Health AI — Flask Server")
    print("=" * 40)
    print(f"Diseases : {len(model.classes_)}")
    print(f"Symptoms : {len(symptom_columns)}")
    print("=" * 40)
    print("Running at http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)