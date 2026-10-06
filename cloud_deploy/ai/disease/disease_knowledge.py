"""
Disease knowledge base for the Smart Agriculture system.

This module maps the AI model's PlantVillage class names to
farmer-readable information.

IMPORTANT:
This is a knowledge/recommendation layer, not a disease classifier.
The ML model determines the predicted disease.
This module only provides supporting information.
"""


# ============================================================
# DISEASE KNOWLEDGE BASE
# ============================================================

DISEASE_KNOWLEDGE = {

    # --------------------------------------------------------
    # APPLE
    # --------------------------------------------------------

    "Apple___Apple_scab": {
        "crop": "Apple",
        "disease": "Apple Scab",
        "type": "Fungal disease",
        "severity": "Moderate to High",
        "symptoms": [
            "Olive or brown spots on leaves",
            "Dark lesions may develop on fruit",
            "Severe infection can cause premature leaf drop"
        ],
        "management": [
            "Remove and destroy infected fallen leaves",
            "Improve air circulation through pruning",
            "Avoid prolonged leaf wetness",
            "Follow locally approved fungicide recommendations"
        ]
    },

    "Apple___Black_rot": {
        "crop": "Apple",
        "disease": "Apple Black Rot",
        "type": "Fungal disease",
        "severity": "Moderate to High",
        "symptoms": [
            "Brown or purple leaf spots",
            "Circular lesions with darker margins",
            "Fruit may develop dark rotting areas"
        ],
        "management": [
            "Remove infected plant material",
            "Prune affected branches",
            "Maintain good orchard sanitation",
            "Use locally approved disease-management practices"
        ]
    },

    "Apple___Cedar_apple_rust": {
        "crop": "Apple",
        "disease": "Apple Cedar Rust",
        "type": "Fungal disease",
        "severity": "Moderate",
        "symptoms": [
            "Yellow-orange spots on leaves",
            "Spots may develop dark centers",
            "Premature leaf loss can occur"
        ],
        "management": [
            "Remove heavily infected leaves",
            "Improve orchard sanitation",
            "Manage nearby alternate hosts where appropriate",
            "Follow local fungicide recommendations"
        ]
    },

    "Apple___healthy": {
        "crop": "Apple",
        "disease": "Healthy",
        "type": "Healthy plant",
        "severity": "None",
        "symptoms": [
            "No major visible disease symptoms detected"
        ],
        "management": [
            "Continue regular crop monitoring",
            "Maintain appropriate irrigation and nutrition",
            "Inspect plants regularly for new symptoms"
        ]
    },


    # --------------------------------------------------------
    # BLUEBERRY
    # --------------------------------------------------------

    "Blueberry___healthy": {
        "crop": "Blueberry",
        "disease": "Healthy",
        "type": "Healthy plant",
        "severity": "None",
        "symptoms": [
            "No major visible disease symptoms detected"
        ],
        "management": [
            "Continue regular crop monitoring",
            "Maintain suitable irrigation and nutrition"
        ]
    },


    # --------------------------------------------------------
    # CHERRY
    # --------------------------------------------------------

    "Cherry_(including_sour)___healthy": {
        "crop": "Cherry",
        "disease": "Healthy",
        "type": "Healthy plant",
        "severity": "None",
        "symptoms": [
            "No major visible disease symptoms detected"
        ],
        "management": [
            "Continue regular monitoring",
            "Maintain suitable irrigation and nutrition"
        ]
    },

    "Cherry_(including_sour)___Powdery_mildew": {
        "crop": "Cherry",
        "disease": "Cherry Powdery Mildew",
        "type": "Fungal disease",
        "severity": "Moderate",
        "symptoms": [
            "White powder-like growth on leaves",
            "Distorted or curled leaves",
            "Young shoots may show fungal growth"
        ],
        "management": [
            "Improve air circulation",
            "Remove severely affected plant material",
            "Avoid excessive humidity around foliage",
            "Follow locally approved fungicide recommendations"
        ]
    },


    # --------------------------------------------------------
    # CORN
    # --------------------------------------------------------

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "crop": "Corn",
        "disease": "Corn Gray Leaf Spot",
        "type": "Fungal disease",
        "severity": "Moderate to High",
        "symptoms": [
            "Long rectangular gray or tan lesions",
            "Lesions develop mainly on leaves",
            "Severe infection can reduce photosynthesis"
        ],
        "management": [
            "Use crop rotation where appropriate",
            "Remove or manage crop residue according to local practice",
            "Monitor fields regularly",
            "Follow locally approved fungicide recommendations"
        ]
    },

    "Corn_(maize)___Common_rust_": {
        "crop": "Corn",
        "disease": "Corn Common Rust",
        "type": "Fungal disease",
        "severity": "Moderate",
        "symptoms": [
            "Small reddish-brown rust pustules",
            "Pustules occur on leaf surfaces",
            "Heavy infection can reduce leaf function"
        ],
        "management": [
            "Monitor plants during favorable weather",
            "Use resistant varieties where available",
            "Maintain appropriate crop management",
            "Follow local disease-control recommendations"
        ]
    },

    "Corn_(maize)___healthy": {
        "crop": "Corn",
        "disease": "Healthy",
        "type": "Healthy plant",
        "severity": "None",
        "symptoms": [
            "No major visible disease symptoms detected"
        ],
        "management": [
            "Continue regular crop monitoring",
            "Maintain suitable irrigation and nutrition"
        ]
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop": "Corn",
        "disease": "Corn Northern Leaf Blight",
        "type": "Fungal disease",
        "severity": "High",
        "symptoms": [
            "Long gray-green or tan leaf lesions",
            "Lesions may become cigar-shaped",
            "Severe infection can cause extensive leaf damage"
        ],
        "management": [
            "Use resistant varieties where available",
            "Practice crop rotation",
            "Manage crop residue appropriately",
            "Follow locally approved disease-management practices"
        ]
    },


    # --------------------------------------------------------
    # GRAPE
    # --------------------------------------------------------

    "Grape___Black_rot": {
        "crop": "Grape",
        "disease": "Grape Black Rot",
        "type": "Fungal disease",
        "severity": "High",
        "symptoms": [
            "Brown circular leaf lesions",
            "Dark fruit lesions",
            "Infected fruit may shrivel into black mummies"
        ],
        "management": [
            "Remove infected fruit and plant debris",
            "Improve canopy ventilation",
            "Avoid prolonged moisture on foliage",
            "Follow locally approved fungicide recommendations"
        ]
    },

    "Grape___Esca_(Black_Measles)": {
        "crop": "Grape",
        "disease": "Grape Esca",
        "type": "Fungal disease",
        "severity": "High",
        "symptoms": [
            "Leaf discoloration and spotting",
            "Interveinal tissue may become damaged",
            "Affected vines may show reduced vigor"
        ],
        "management": [
            "Remove severely affected plant material",
            "Maintain vineyard sanitation",
            "Monitor affected vines closely",
            "Follow local vineyard disease-management practices"
        ]
    },

    "Grape___healthy": {
        "crop": "Grape",
        "disease": "Healthy",
        "type": "Healthy plant",
        "severity": "None",
        "symptoms": [
            "No major visible disease symptoms detected"
        ],
        "management": [
            "Continue regular vineyard monitoring",
            "Maintain appropriate irrigation and nutrition"
        ]
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "crop": "Grape",
        "disease": "Grape Leaf Blight",
        "type": "Fungal disease",
        "severity": "Moderate",
        "symptoms": [
            "Brown or dark leaf spots",
            "Spots may enlarge under favorable conditions",
            "Affected leaves can decline prematurely"
        ],
        "management": [
            "Remove severely affected leaves",
            "Improve canopy airflow",
            "Avoid unnecessary leaf wetness",
            "Follow local disease-management recommendations"
        ]
    },


    # --------------------------------------------------------
    # ORANGE
    # --------------------------------------------------------

    "Orange___Haunglongbing_(Citrus_greening)": {
        "crop": "Orange",
        "disease": "Citrus Greening",
        "type": "Bacterial disease",
        "severity": "High",
        "symptoms": [
            "Uneven yellowing of leaves",
            "Leaf veins may remain greener than surrounding tissue",
            "Fruit may be small or develop uneven coloration"
        ],
        "management": [
            "Monitor trees regularly",
            "Manage insect vectors according to local agricultural guidance",
            "Remove severely affected trees where recommended",
            "Use certified healthy planting material"
        ]
    },


    # --------------------------------------------------------
    # PEACH
    # --------------------------------------------------------

    "Peach___Bacterial_spot": {
        "crop": "Peach",
        "disease": "Peach Bacterial Spot",
        "type": "Bacterial disease",
        "severity": "Moderate",
        "symptoms": [
            "Small dark spots on leaves",
            "Leaf tissue may become damaged around lesions",
            "Fruit may develop sunken spots"
        ],
        "management": [
            "Maintain good orchard sanitation",
            "Avoid unnecessary leaf wetness",
            "Use suitable resistant varieties where available",
            "Follow local bacterial-disease management guidance"
        ]
    },

    "Peach___healthy": {
        "crop": "Peach",
        "disease": "Healthy",
        "type": "Healthy plant",
        "severity": "None",
        "symptoms": [
            "No major visible disease symptoms detected"
        ],
        "management": [
            "Continue regular monitoring",
            "Maintain suitable irrigation and nutrition"
        ]
    },


    # --------------------------------------------------------
    # PEPPER
    # --------------------------------------------------------

    "Pepper,_bell___Bacterial_spot": {
        "crop": "Bell Pepper",
        "disease": "Bacterial Spot",
        "type": "Bacterial disease",
        "severity": "Moderate to High",
        "symptoms": [
            "Small dark spots on leaves",
            "Lesions may enlarge under favorable conditions",
            "Fruit may develop raised or scab-like lesions"
        ],
        "management": [
            "Use disease-free seed or planting material",
            "Avoid working with wet plants",
            "Improve field sanitation",
            "Follow locally approved bacterial disease-management practices"
        ]
    },

    "Pepper,_bell___healthy": {
        "crop": "Bell Pepper",
        "disease": "Healthy",
        "type": "Healthy plant",
        "severity": "None",
        "symptoms": [
            "No major visible disease symptoms detected"
        ],
        "management": [
            "Continue regular crop monitoring",
            "Maintain suitable irrigation and nutrition"
        ]
    },


    # --------------------------------------------------------
    # POTATO
    # --------------------------------------------------------

    "Potato___Early_blight": {
        "crop": "Potato",
        "disease": "Potato Early Blight",
        "type": "Fungal disease",
        "severity": "Moderate to High",
        "symptoms": [
            "Brown circular leaf lesions",
            "Concentric ring patterns may appear",
            "Older leaves are often affected first"
        ],
        "management": [
            "Remove severely infected plant material",
            "Maintain appropriate plant nutrition",
            "Avoid prolonged leaf wetness",
            "Follow locally approved fungicide recommendations"
        ]
    },

    "Potato___healthy": {
        "crop": "Potato",
        "disease": "Healthy",
        "type": "Healthy plant",
        "severity": "None",
        "symptoms": [
            "No major visible disease symptoms detected"
        ],
        "management": [
            "Continue regular monitoring",
            "Maintain suitable irrigation and nutrition"
        ]
    },

    "Potato___Late_blight": {
        "crop": "Potato",
        "disease": "Potato Late Blight",
        "type": "Oomycete disease",
        "severity": "High",
        "symptoms": [
            "Dark water-soaked leaf lesions",
            "Rapid expansion of lesions under favorable conditions",
            "Severe infection can destroy foliage"
        ],
        "management": [
            "Monitor crops frequently during cool and wet conditions",
            "Remove severely infected plant material where appropriate",
            "Avoid prolonged leaf wetness",
            "Follow locally approved late-blight management recommendations"
        ]
    },


    # --------------------------------------------------------
    # RASPBERRY
    # --------------------------------------------------------

    "Raspberry___healthy": {
        "crop": "Raspberry",
        "disease": "Healthy",
        "type": "Healthy plant",
        "severity": "None",
        "symptoms": [
            "No major visible disease symptoms detected"
        ],
        "management": [
            "Continue regular crop monitoring",
            "Maintain suitable irrigation and nutrition"
        ]
    },


    # --------------------------------------------------------
    # SOYBEAN
    # --------------------------------------------------------

    "Soybean___healthy": {
        "crop": "Soybean",
        "disease": "Healthy",
        "type": "Healthy plant",
        "severity": "None",
        "symptoms": [
            "No major visible disease symptoms detected"
        ],
        "management": [
            "Continue regular crop monitoring",
            "Maintain suitable irrigation and nutrition"
        ]
    },


    # --------------------------------------------------------
    # SQUASH
    # --------------------------------------------------------

    "Squash___Powdery_mildew": {
        "crop": "Squash",
        "disease": "Squash Powdery Mildew",
        "type": "Fungal disease",
        "severity": "Moderate",
        "symptoms": [
            "White powder-like growth on leaves",
            "Leaves may become yellow or weakened",
            "Severe infection can reduce photosynthesis"
        ],
        "management": [
            "Improve air circulation",
            "Avoid excessive humidity around foliage",
            "Remove severely affected leaves where appropriate",
            "Follow locally approved disease-control recommendations"
        ]
    },


    # --------------------------------------------------------
    # STRAWBERRY
    # --------------------------------------------------------

    "Strawberry___healthy": {
        "crop": "Strawberry",
        "disease": "Healthy",
        "type": "Healthy plant",
        "severity": "None",
        "symptoms": [
            "No major visible disease symptoms detected"
        ],
        "management": [
            "Continue regular crop monitoring",
            "Maintain appropriate irrigation and nutrition"
        ]
    },

    "Strawberry___Leaf_scorch": {
        "crop": "Strawberry",
        "disease": "Strawberry Leaf Scorch",
        "type": "Fungal disease",
        "severity": "Moderate",
        "symptoms": [
            "Small dark purple or brown leaf spots",
            "Spots may enlarge and cause leaf tissue to dry",
            "Severe infection can reduce plant vigor"
        ],
        "management": [
            "Remove severely affected leaves",
            "Improve field sanitation",
            "Avoid prolonged leaf wetness",
            "Follow local disease-management recommendations"
        ]
    },


    # --------------------------------------------------------
    # TOMATO
    # --------------------------------------------------------

    "Tomato___Bacterial_spot": {
        "crop": "Tomato",
        "disease": "Tomato Bacterial Spot",
        "type": "Bacterial disease",
        "severity": "Moderate to High",
        "symptoms": [
            "Small dark spots on leaves",
            "Lesions can become surrounded by yellow tissue",
            "Fruit may develop small raised spots"
        ],
        "management": [
            "Use disease-free seed and planting material",
            "Avoid handling plants when foliage is wet",
            "Remove severely infected plant material",
            "Follow locally approved bacterial disease-management practices"
        ]
    },

    "Tomato___Early_blight": {
        "crop": "Tomato",
        "disease": "Tomato Early Blight",
        "type": "Fungal disease",
        "severity": "Moderate to High",
        "symptoms": [
            "Brown circular lesions",
            "Concentric rings may form inside lesions",
            "Lower leaves are commonly affected first"
        ],
        "management": [
            "Remove severely affected leaves",
            "Maintain good plant spacing and airflow",
            "Avoid prolonged leaf wetness",
            "Follow locally approved fungicide recommendations"
        ]
    },

    "Tomato___healthy": {
        "crop": "Tomato",
        "disease": "Healthy",
        "type": "Healthy plant",
        "severity": "None",
        "symptoms": [
            "No major visible disease symptoms detected"
        ],
        "management": [
            "Continue regular crop monitoring",
            "Maintain appropriate irrigation and nutrition"
        ]
    },

    "Tomato___Late_blight": {
        "crop": "Tomato",
        "disease": "Tomato Late Blight",
        "type": "Oomycete disease",
        "severity": "High",
        "symptoms": [
            "Dark water-soaked lesions",
            "Lesions can spread rapidly",
            "Leaves may become brown and collapse"
        ],
        "management": [
            "Monitor crops frequently during cool and wet conditions",
            "Remove severely infected plant material where appropriate",
            "Avoid prolonged leaf wetness",
            "Follow locally approved late-blight management recommendations"
        ]
    },

    "Tomato___Leaf_Mold": {
        "crop": "Tomato",
        "disease": "Tomato Leaf Mold",
        "type": "Fungal disease",
        "severity": "Moderate",
        "symptoms": [
            "Yellow patches may develop on upper leaf surfaces",
            "Olive or gray fungal growth may appear on leaf undersides",
            "Leaves may dry and decline in severe cases"
        ],
        "management": [
            "Improve ventilation around plants",
            "Reduce excessive humidity",
            "Avoid prolonged leaf wetness",
            "Remove severely affected leaves",
            "Follow locally approved disease-management practices"
        ]
    },

    "Tomato___Septoria_leaf_spot": {
        "crop": "Tomato",
        "disease": "Tomato Septoria Leaf Spot",
        "type": "Fungal disease",
        "severity": "Moderate",
        "symptoms": [
            "Small circular leaf spots",
            "Spots may have dark margins and lighter centers",
            "Lower leaves are often affected first"
        ],
        "management": [
            "Remove affected leaves",
            "Improve airflow",
            "Avoid overhead irrigation where possible",
            "Maintain field sanitation",
            "Follow locally approved fungicide recommendations"
        ]
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "crop": "Tomato",
        "disease": "Two-Spotted Spider Mite Damage",
        "type": "Pest",
        "severity": "Moderate to High",
        "symptoms": [
            "Fine speckling or stippling on leaves",
            "Leaves may become yellow or bronze",
            "Fine webbing may appear during heavy infestation"
        ],
        "management": [
            "Inspect the undersides of leaves",
            "Monitor pest population regularly",
            "Encourage appropriate biological control where available",
            "Follow locally approved pest-management recommendations"
        ]
    },

    "Tomato___Target_Spot": {
        "crop": "Tomato",
        "disease": "Tomato Target Spot",
        "type": "Fungal disease",
        "severity": "Moderate",
        "symptoms": [
            "Circular brown lesions",
            "Concentric rings may give a target-like appearance",
            "Leaves may yellow and decline"
        ],
        "management": [
            "Remove severely affected leaves",
            "Improve plant airflow",
            "Avoid prolonged leaf wetness",
            "Follow locally approved fungicide recommendations"
        ]
    },

    "Tomato___Tomato_mosaic_virus": {
        "crop": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "type": "Viral disease",
        "severity": "High",
        "symptoms": [
            "Mosaic pattern of light and dark green areas",
            "Leaf distortion may occur",
            "Plant growth may be reduced"
        ],
        "management": [
            "Remove severely infected plants where appropriate",
            "Disinfect tools after handling infected plants",
            "Use clean planting material",
            "Control spread through good sanitation"
        ]
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "type": "Viral disease",
        "severity": "High",
        "symptoms": [
            "Leaf curling",
            "Yellowing of leaves",
            "Stunted plant growth",
            "Reduced fruit production"
        ],
        "management": [
            "Monitor and manage whitefly vectors according to local guidance",
            "Remove severely affected plants where appropriate",
            "Use healthy planting material",
            "Maintain field sanitation"
        ]
    }
}


# ============================================================
# DEFAULT INFORMATION
# ============================================================

DEFAULT_DISEASE_INFO = {
    "crop": "Unknown",
    "disease": "Unknown",
    "type": "Unknown",
    "severity": "Unknown",
    "symptoms": [
        "No disease-specific information available."
    ],
    "management": [
        "Consult a qualified agricultural expert.",
        "Do not apply treatment solely from an AI prediction."
    ]
}


# ============================================================
# GET DISEASE INFORMATION
# ============================================================

def get_disease_info(class_name):
    """
    Return information associated with a model class.
    """

    return DISEASE_KNOWLEDGE.get(
        class_name,
        DEFAULT_DISEASE_INFO
    )


# ============================================================
# ADD KNOWLEDGE TO PREDICTION
# ============================================================

def enrich_prediction(prediction):
    """
    Add disease knowledge to a model prediction dictionary.
    """

    class_name = prediction.get("class_name")

    information = get_disease_info(
        class_name
    )

    return {
        **prediction,
        "crop": information["crop"],
        "disease": information["disease"],
        "type": information["type"],
        "severity": information["severity"],
        "symptoms": information["symptoms"],
        "management": information["management"]
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_class = "Tomato___Late_blight"

    information = get_disease_info(
        test_class
    )

    print("=" * 60)
    print("DISEASE KNOWLEDGE TEST")
    print("=" * 60)

    print(f"\nClass: {test_class}")

    print(
        f"Crop: {information['crop']}"
    )

    print(
        f"Disease: {information['disease']}"
    )

    print(
        f"Type: {information['type']}"
    )

    print(
        f"Severity: {information['severity']}"
    )

    print("\nSymptoms:")

    for symptom in information["symptoms"]:
        print(f"- {symptom}")

    print("\nManagement:")

    for recommendation in information["management"]:
        print(f"- {recommendation}")

    print("\nKnowledge base test completed.")