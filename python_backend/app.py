# Heavy Metal Pollution Indices Analysis Application
import os
import json
import signal
import sys
from typing import Dict, Any, List, Tuple
from functools import lru_cache
import gc
import numpy as np

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import folium
from pymongo import MongoClient

# Logging configuration
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global MongoDB client
_mongo_client = None

# Heavy metal background reference values (mg/kg) - typical soil background values
BACKGROUND_VALUES = {
    'Cd': 0.3,    # Cadmium
    'Pb': 20,     # Lead
    'Cr': 50,     # Chromium
    'Cu': 30,     # Copper
    'Zn': 90,     # Zinc
    'Ni': 40,     # Nickel
    'As': 10,     # Arsenic
    'Hg': 0.05,   # Mercury
    'Co': 15,     # Cobalt
    'Mn': 600     # Manganese
}

# Toxicity response factors for different metals
TOXIC_RESPONSE_FACTORS = {
    'Cd': 30,
    'Pb': 5,
    'Cr': 2,
    'Cu': 5,
    'Zn': 1,
    'Ni': 5,
    'As': 10,
    'Hg': 40,
    'Co': 5,
    'Mn': 1
}

def get_mongo_client():
    """Get or create MongoDB client connection"""
    global _mongo_client
    if _mongo_client is None:
        uri = os.environ.get("MONGO_URI") or "mongodb+srv://shadowblue976_db_user:iWBGGI4i3chxfTXW@metalsense.ldwosxe.mongodb.net/metal/Disease_Data/"
        if uri.endswith('/'):
            uri = uri[:-1]
        _mongo_client = MongoClient(uri, maxPoolSize=10, minPoolSize=1)
    return _mongo_client

def calculate_contamination_factor(metal_conc: float, background_value: float) -> float:
    """Calculate Contamination Factor (CF) for a single metal"""
    if background_value == 0:
        return 0
    return metal_conc / background_value

def calculate_pollution_load_index(contamination_factors: List[float]) -> float:
    """Calculate Pollution Load Index (PLI)"""
    if not contamination_factors or any(cf <= 0 for cf in contamination_factors):
        return 0
    product = np.prod(contamination_factors)
    return product ** (1 / len(contamination_factors))

def calculate_enrichment_factor(metal_conc: float, ref_conc: float, 
                                 metal_bg: float, ref_bg: float) -> float:
    """Calculate Enrichment Factor (EF) using reference element (usually Fe or Al)"""
    if metal_bg == 0 or ref_bg == 0:
        return 0
    return (metal_conc / ref_conc) / (metal_bg / ref_bg)

def calculate_geoaccumulation_index(metal_conc: float, background_value: float) -> float:
    """Calculate Geo-accumulation Index (Igeo)"""
    if background_value == 0:
        return 0
    return np.log2(metal_conc / (1.5 * background_value))

def calculate_ecological_risk_index(metal_conc: float, background_value: float, 
                                      toxic_factor: float) -> float:
    """Calculate Potential Ecological Risk Index (RI)"""
    cf = calculate_contamination_factor(metal_conc, background_value)
    return toxic_factor * cf

def calculate_nemerow_index(contamination_factors: List[float]) -> float:
    """Calculate Nemerow Pollution Index (NPI)"""
    if not contamination_factors:
        return 0
    cf_avg = np.mean(contamination_factors)
    cf_max = np.max(contamination_factors)
    return np.sqrt((cf_avg**2 + cf_max**2) / 2)

def calculate_heavy_metal_evaluation_index(concentrations: Dict[str, float], 
                                             standards: Dict[str, float]) -> float:
    """Calculate Heavy Metal Evaluation Index (HEI)"""
    if not concentrations or not standards:
        return 0
    hei_sum = sum(concentrations[metal] / standards[metal] 
                  for metal in concentrations if metal in standards and standards[metal] > 0)
    return hei_sum

def classify_pollution_level(pli: float, npi: float, hei: float) -> Tuple[str, str]:
    """
    Classify pollution level based on multiple indices
    Returns: (risk_band, risk_level_description)
    """
    # PLI Classification: <1 (no pollution), 1-2 (moderate), 2-3 (heavy), >3 (extreme)
    # NPI Classification: <0.7 (clean), 0.7-1 (warning), 1-2 (light), 2-3 (moderate), >3 (heavy)
    # HEI Classification: <10 (low), 10-20 (moderate), >20 (high)
    
    risk_score = 0
    
    if pli >= 3:
        risk_score += 40
    elif pli >= 2:
        risk_score += 30
    elif pli >= 1:
        risk_score += 20
    else:
        risk_score += 10
    
    if npi >= 3:
        risk_score += 30
    elif npi >= 2:
        risk_score += 20
    elif npi >= 1:
        risk_score += 15
    else:
        risk_score += 5
    
    if hei >= 20:
        risk_score += 30
    elif hei >= 10:
        risk_score += 15
    else:
        risk_score += 5
    
    # Normalize to percentage (max possible score is 100)
    risk_percentage = min(risk_score, 100)
    
    if risk_percentage >= 70:
        return 'red', 'Severe Pollution'
    elif risk_percentage >= 50:
        return 'yellow', 'Moderate Pollution'
    else:
        return 'green', 'Low Pollution'

def band_from_percent(p: float) -> str:
    """Classify risk band from percentage"""
    if p >= 70:
        return 'red'
    if p >= 50:
        return 'yellow'
    return 'green'

@lru_cache(maxsize=32)
def compute_thresholds_cached(indices_tuple) -> Dict[str, float]:
    """Cached threshold computation for pollution indices"""
    thresholds = {}
    for index_name, values_tuple in indices_tuple:
        try:
            vals = pd.Series(list(values_tuple)).fillna(0)
            q75 = float(vals.quantile(0.75))
            thresholds[str(index_name)] = max(q75, 50.0)
        except Exception:
            thresholds[str(index_name)] = 50.0
    return thresholds

def compute_payload() -> Dict[str, Any]:
    """Main computation function for metal pollution analysis"""
    try:
        logger.info("Starting metal pollution analysis")
        
        client = get_mongo_client()
        try:
            db = client.get_default_database()
        except Exception:
            db = None
        if db is None:
            db_name = os.environ.get("MONGO_DB") or "metal_pollution"
            db = client[db_name]

        coll_name = os.environ.get('MONGO_COLL', 'metal_data')
        col = db[coll_name]

        # Essential fields for metal pollution data
        essential_fields = {
            "state": 1,
            "district": 1,
            "location": 1,
            "latitude": 1,
            "longitude": 1,
            "sample_date": 1,
            "year": 1,
            "month": 1,
            # Metal concentrations
            "Cd": 1, "Pb": 1, "Cr": 1, "Cu": 1, "Zn": 1,
            "Ni": 1, "As": 1, "Hg": 1, "Co": 1, "Mn": 1,
            "Fe": 1, "Al": 1  # Reference elements
        }

        cursor = col.find({}, essential_fields).limit(10000)
        docs = list(cursor)
        
        logger.info(f"Retrieved {len(docs)} metal pollution samples")

        if not docs:
            return {"areas": [], "pollution_zones": [], "mapPath": None, "statistics": {}}

        all_results = []
        chunk_size = 1000
        
        for i in range(0, len(docs), chunk_size):
            chunk_docs = docs[i:i + chunk_size]
            df = pd.DataFrame(chunk_docs)
            
            # Normalize numeric columns
            metal_columns = ['Cd', 'Pb', 'Cr', 'Cu', 'Zn', 'Ni', 'As', 'Hg', 'Co', 'Mn', 'Fe', 'Al']
            for col in metal_columns + ['latitude', 'longitude', 'year', 'month']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill missing location data
            for col in ['state', 'district', 'location']:
                if col not in df.columns:
                    df[col] = ""
            
            # Calculate pollution indices for each sample
            for idx, row in df.iterrows():
                # Extract metal concentrations
                metals = {}
                contamination_factors = []
                ecological_risks = []
                
                for metal in BACKGROUND_VALUES.keys():
                    if metal in row and pd.notnull(row[metal]):
                        conc = float(row[metal])
                        metals[metal] = conc
                        
                        # Calculate CF
                        cf = calculate_contamination_factor(conc, BACKGROUND_VALUES[metal])
                        contamination_factors.append(cf)
                        
                        # Calculate Ecological Risk
                        if metal in TOXIC_RESPONSE_FACTORS:
                            er = calculate_ecological_risk_index(
                                conc, BACKGROUND_VALUES[metal], TOXIC_RESPONSE_FACTORS[metal]
                            )
                            ecological_risks.append(er)
                
                if not contamination_factors:
                    continue
                
                # Calculate indices
                pli = calculate_pollution_load_index(contamination_factors)
                npi = calculate_nemerow_index(contamination_factors)
                hei = calculate_heavy_metal_evaluation_index(metals, BACKGROUND_VALUES)
                total_er = sum(ecological_risks) if ecological_risks else 0
                
                # Classify pollution level
                risk_band, risk_description = classify_pollution_level(pli, npi, hei)
                
                # Calculate risk percentage (0-100)
                risk_percentage = min((pli * 20 + npi * 15 + hei * 2 + total_er * 0.5), 100)
                
                # Prepare result
                sample_date = row.get('sample_date', '')
                if pd.isnull(sample_date) or sample_date == '':
                    year = row.get('year', '')
                    month = row.get('month', '')
                    if pd.notnull(year):
                        sample_date = f"{int(year)}-{int(month):02d}" if pd.notnull(month) else f"{int(year)}"
                
                result = {
                    "state": str(row.get("state", "") or ""),
                    "district": str(row.get('district', '') or ''),
                    "location": str(row.get("location", "") or ""),
                    "date": str(sample_date),
                    "lat": safe_float(row.get('latitude')),
                    "lon": safe_float(row.get('longitude')),
                    "indices": {
                        "PLI": round(pli, 3),
                        "NPI": round(npi, 3),
                        "HEI": round(hei, 3),
                        "TotalER": round(total_er, 3)
                    },
                    "risk_percentage": round(risk_percentage, 2),
                    "risk_band": risk_band,
                    "risk_level": risk_description,
                    "metal_concentrations": {k: round(v, 3) for k, v in metals.items()},
                    "contamination_factors": {
                        metal: round(calculate_contamination_factor(metals[metal], BACKGROUND_VALUES[metal]), 3)
                        for metal in metals if metal in BACKGROUND_VALUES
                    }
                }
                
                all_results.append(result)
            
            del df
            gc.collect()

        # Sort by risk percentage
        all_results.sort(key=lambda x: x["risk_percentage"], reverse=True)

        # Generate map
        map_url_path = None
        try:
            m = folium.Map(location=[22.9734, 78.6569], zoom_start=5)
            
            # Plot top 500 samples
            plot_data = [r for r in all_results[:500] if r['lat'] is not None and r['lon'] is not None]
            
            for result in plot_data:
                band = result['risk_band']
                color = 'red' if band == 'red' else ('yellow' if band == 'yellow' else 'green')
                
                indices = result['indices']
                popup_text = (
                    f"<b>State:</b> {result['state']}<br>"
                    f"<b>District:</b> {result['district']}<br>"
                    f"<b>Location:</b> {result['location']}<br>"
                    f"<b>Date:</b> {result['date']}<br>"
                    f"<b>Risk Level:</b> {result['risk_level']}<br>"
                    f"<b>Risk %:</b> {result['risk_percentage']}%<br>"
                    f"<hr>"
                    f"<b>PLI:</b> {indices['PLI']}<br>"
                    f"<b>NPI:</b> {indices['NPI']}<br>"
                    f"<b>HEI:</b> {indices['HEI']}<br>"
                    f"<b>Total ER:</b> {indices['TotalER']}"
                )
                
                folium.CircleMarker(
                    location=[result['lat'], result['lon']],
                    radius=8,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=popup_text
                ).add_to(m)

            static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
            os.makedirs(static_dir, exist_ok=True)
            map_path = os.path.join(static_dir, 'india_metal_pollution_map.html')
            m.save(map_path)
            map_url_path = '/static/india_metal_pollution_map.html'
            
        except Exception as e:
            logger.error(f"Map generation error: {e}")

        # Calculate summary statistics
        if all_results:
            stats = {
                "total_samples": len(all_results),
                "severe_pollution_sites": len([r for r in all_results if r['risk_band'] == 'red']),
                "moderate_pollution_sites": len([r for r in all_results if r['risk_band'] == 'yellow']),
                "low_pollution_sites": len([r for r in all_results if r['risk_band'] == 'green']),
                "average_pli": round(np.mean([r['indices']['PLI'] for r in all_results]), 3),
                "average_risk_percentage": round(np.mean([r['risk_percentage'] for r in all_results]), 2)
            }
        else:
            stats = {}

        gc.collect()
        
        logger.info(f"Analysis completed with {len(all_results)} samples")
        return {
            "areas": all_results,
            "pollution_zones": all_results,
            "mapPath": map_url_path,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error in compute_payload: {e}")
        return {"areas": [], "pollution_zones": [], "mapPath": None, "statistics": {}, "error": str(e)}

def safe_float(x):
    """Safely convert to float"""
    try:
        return float(x)
    except Exception:
        return None

# Flask application
app = Flask(__name__)
CORS(app, resources={r"/api/": {"origins": "*"}})

@app.route('/static/<filename>')
def static_files(filename):
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    return send_from_directory(static_dir, filename)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"ok": True, "status": "healthy"})

@app.route('/api/analyze', methods=['GET'])
def analyze():
    """Main API endpoint for metal pollution analysis"""
    try:
        logger.info("Received metal pollution analysis request")
        payload = compute_payload()
        logger.info("Sending analysis response")
        return jsonify(payload)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({
            "pollution_zones": [], 
            "areas": [], 
            "mapPath": None, 
            "statistics": {},
            "error": str(e)
        }), 500

@app.route('/api/indices-info', methods=['GET'])
def indices_info():
    """Endpoint to get information about pollution indices"""
    info = {
        "PLI": {
            "name": "Pollution Load Index",
            "description": "Geometric mean of contamination factors",
            "interpretation": {
                "<1": "No pollution",
                "1-2": "Moderate pollution",
                "2-3": "Heavy pollution",
                ">3": "Extreme pollution"
            }
        },
        "NPI": {
            "name": "Nemerow Pollution Index",
            "description": "Comprehensive pollution index considering average and maximum CF",
            "interpretation": {
                "<0.7": "Clean",
                "0.7-1": "Warning limit",
                "1-2": "Light pollution",
                "2-3": "Moderate pollution",
                ">3": "Heavy pollution"
            }
        },
        "HEI": {
            "name": "Heavy Metal Evaluation Index",
            "description": "Sum of ratios of metal concentrations to standards",
            "interpretation": {
                "<10": "Low pollution",
                "10-20": "Moderate pollution",
                ">20": "High pollution"
            }
        },
        "ER": {
            "name": "Ecological Risk Index",
            "description": "Potential ecological risk from toxic metals",
            "interpretation": {
                "<40": "Low risk",
                "40-80": "Moderate risk",
                "80-160": "Considerable risk",
                "160-320": "High risk",
                ">320": "Very high risk"
            }
        }
    }
    return jsonify(info)

def signal_handler(sig, frame):
    logger.info('Gracefully shutting down Flask server')
    global _mongo_client
    if _mongo_client:
        _mongo_client.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT') or os.environ.get('PORT', '5000'))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    
    logger.info(f"Starting Heavy Metal Pollution Analysis Server on {host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)