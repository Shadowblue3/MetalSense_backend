# Heavy Metal Pollution Indices Analysis API - Production Ready (Disease_Data integrated)
import os
import json
import signal
import sys
from typing import Dict, Any, List, Tuple
from functools import lru_cache
from datetime import datetime
import gc
import numpy as np
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, jsonify, send_from_directory, request, make_response
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
# Cache for last generated map HTML (served at /api/map)
_last_map_html = None

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
    'Cd': 30, 'Pb': 5, 'Cr': 2, 'Cu': 5, 'Zn': 1,
    'Ni': 5, 'As': 10, 'Hg': 40, 'Co': 5, 'Mn': 1
}


def get_mongo_client():
    """Get or create MongoDB client connection"""
    global _mongo_client
    if _mongo_client is None:
        uri = os.environ.get("MONGO_URI") or "mongodb+srv://shadowblue976_db_user:iWBGGI4i3chxfTXW@metalsense.ldwosxe.mongodb.net/metal"
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


def calculate_ecological_risk_index(metal_conc: float, background_value: float,
                                      toxic_factor: float) -> float:
    """Calculate Potential Ecological Risk Index (RI)"""
    cf = calculate_contamination_factor(metal_conc, background_value)
    return toxic_factor * cf


def classify_pollution_level(pli: float, npi: float, hei: float) -> Tuple[str, str, float]:
    """
    Classify pollution level based on multiple indices
    Returns: (risk_band, risk_level_description, risk_percentage)
    """
    risk_score = 0

    # PLI scoring
    if pli >= 3:
        risk_score += 40
    elif pli >= 2:
        risk_score += 30
    elif pli >= 1:
        risk_score += 20
    else:
        risk_score += 10

    # NPI scoring
    if npi >= 3:
        risk_score += 30
    elif npi >= 2:
        risk_score += 20
    elif npi >= 1:
        risk_score += 15
    else:
        risk_score += 5

    # HEI scoring
    if hei >= 20:
        risk_score += 30
    elif hei >= 10:
        risk_score += 15
    else:
        risk_score += 5

    risk_percentage = min(risk_score, 100)

    if risk_percentage >= 70:
        return 'red', 'Severe Pollution', risk_percentage
    elif risk_percentage >= 50:
        return 'yellow', 'Moderate Pollution', risk_percentage
    else:
        return 'green', 'Low Pollution', risk_percentage


def safe_float(x):
    """Safely convert to float"""
    try:
        return float(x)
    except Exception:
        return None


def generate_chart_data(all_results: List[Dict]) -> Dict[str, Any]:
    """Generate data for charts and graphs"""
    if not all_results:
        return {}

    # Index distribution data
    pli_values = [r['indices']['PLI'] for r in all_results]
    npi_values = [r['indices']['NPI'] for r in all_results]
    hei_values = [r['indices']['HEI'] for r in all_results]

    # State-wise statistics
    state_stats = {}
    for result in all_results:
        state = result['state']
        if state not in state_stats:
            state_stats[state] = {
                'total_samples': 0,
                'severe_count': 0,
                'moderate_count': 0,
                'low_count': 0,
                'avg_pli': []
            }
        state_stats[state]['total_samples'] += 1
        state_stats[state]['avg_pli'].append(result['indices']['PLI'])

        if result['risk']['band'] == 'red':
            state_stats[state]['severe_count'] += 1
        elif result['risk']['band'] == 'yellow':
            state_stats[state]['moderate_count'] += 1
        else:
            state_stats[state]['low_count'] += 1

    # Calculate averages
    for state in state_stats:
        state_stats[state]['avg_pli'] = round(np.mean(state_stats[state]['avg_pli']), 3)

    # Metal-wise contamination
    metal_contamination = {}
    for metal in BACKGROUND_VALUES.keys():
        metal_cf = [r['contamination_factors'].get(metal, 0) for r in all_results
                    if metal in r['contamination_factors']]
        if metal_cf:
            metal_contamination[metal] = {
                'avg_cf': round(np.mean(metal_cf), 3),
                'max_cf': round(np.max(metal_cf), 3),
                'min_cf': round(np.min(metal_cf), 3)
            }

    # Time series data (if date available)
    time_series = {}
    for result in all_results:
        date = result.get('date', '')
        if date and date != 'N/A':
            if date not in time_series:
                time_series[date] = {'severe': 0, 'moderate': 0, 'low': 0}
            time_series[date][result['risk']['band'].replace('red', 'severe').replace('yellow', 'moderate').replace('green', 'low')] += 1

    return {
        'index_distribution': {
            'pli': {
                'mean': round(np.mean(pli_values), 3),
                'median': round(np.median(pli_values), 3),
                'max': round(np.max(pli_values), 3),
                'min': round(np.min(pli_values), 3)
            },
            'npi': {
                'mean': round(np.mean(npi_values), 3),
                'median': round(np.median(npi_values), 3),
                'max': round(np.max(npi_values), 3),
                'min': round(np.min(npi_values), 3)
            },
            'hei': {
                'mean': round(np.mean(hei_values), 3),
                'median': round(np.median(hei_values), 3),
                'max': round(np.max(hei_values), 3),
                'min': round(np.min(hei_values), 3)
            }
        },
        'state_wise_stats': state_stats,
        'metal_contamination': metal_contamination,
        'time_series': dict(sorted(time_series.items()))
    }


def compute_payload(limit: int = None, state_filter: str = None,
                   district_filter: str = None, min_risk: float = None) -> Dict[str, Any]:
    """Main computation function with filtering options for Disease_Data schema"""
    try:
        logger.info("Starting metal pollution analysis")
        global _last_map_html

        client = get_mongo_client()
        try:
            db = client.get_default_database()
        except Exception:
            db = None
        if db is None:
            db_name = os.environ.get("MONGO_DB") or "metal"
            db = client[db_name]

        # Default to Disease_Data collection as requested
        coll_name = os.environ.get('MONGO_COLL', 'Disease_Data')
        col = db[coll_name]

        # Build query filters (support both `state` and `state_ut` fields)
        query: Dict[str, Any] = {}
        if state_filter:
            query['$or'] = [
                {'state': {'$regex': state_filter, '$options': 'i'}},
                {'state_ut': {'$regex': state_filter, '$options': 'i'}}
            ]
        if district_filter:
            query['district'] = {'$regex': district_filter, '$options': 'i'}

        # Essential fields (support both lowercase and TitleCase geolocation keys)
        essential_fields = {
            "state": 1, "state_ut": 1, "district": 1, "location": 1,
            "latitude": 1, "longitude": 1, "Latitude": 1, "Longitude": 1,
            "sample_date": 1, "year": 1, "month": 1,
            "pH": 1, "EC": 1,
            "As": 1, "Cd": 1, "Cr": 1, "Cu": 1, "Pb": 1, "Zn": 1, "Ni": 1,
            "Hg": 1, "Co": 1, "Mn": 1, "Fe": 1, "Al": 1,
            "Background_As": 1, "Background_Cd": 1, "Background_Cr": 1, "Background_Cu": 1,
            "Background_Pb": 1, "Background_Zn": 1, "Background_Ni": 1,
            "Background_Hg": 1, "Background_Co": 1, "Background_Mn": 1
        }

        query_limit = limit if limit else 10000
        cursor = col.find(query, essential_fields).limit(query_limit)
        docs = list(cursor)

        logger.info(f"Retrieved {len(docs)} metal pollution samples")

        if not docs:
            return {
                "success": True,
                "data": [],
                "mapPath": None,
                "statistics": {},
                "charts": {},
                "message": "No data found matching the criteria"
            }

        all_results: List[Dict[str, Any]] = []
        chunk_size = 1000

        for i in range(0, len(docs), chunk_size):
            chunk_docs = docs[i:i + chunk_size]
            df = pd.DataFrame(chunk_docs)

            # Normalize numeric columns
            metal_columns = ['Cd', 'Pb', 'Cr', 'Cu', 'Zn', 'Ni', 'As', 'Hg', 'Co', 'Mn', 'Fe', 'Al']
            for colname in metal_columns + ['latitude', 'longitude', 'Latitude', 'Longitude', 'year', 'month', 'pH', 'EC'] + [f'Background_{m}' for m in metal_columns]:
                if colname in df.columns:
                    df[colname] = pd.to_numeric(df[colname], errors='coerce')

            for colname in ['state', 'state_ut', 'district', 'location']:
                if colname not in df.columns:
                    df[colname] = ""

            for idx, row in df.iterrows():
                metals: Dict[str, float] = {}
                contamination_factors: List[float] = []
                ecological_risks: List[float] = []

                # Resolve per-row background values
                backgrounds = BACKGROUND_VALUES.copy()
                for metal in BACKGROUND_VALUES.keys():
                    bg_key = f'Background_{metal}'
                    if bg_key in row and pd.notnull(row[bg_key]):
                        try:
                            backgrounds[metal] = float(row[bg_key])
                        except Exception:
                            pass

                # Collect metals for this row
                for metal in BACKGROUND_VALUES.keys():
                    if metal in row and pd.notnull(row[metal]):
                        try:
                            conc = float(row[metal])
                        except Exception:
                            continue
                        metals[metal] = conc
                        bg_val = backgrounds.get(metal, BACKGROUND_VALUES[metal])
                        cf = calculate_contamination_factor(conc, bg_val)
                        contamination_factors.append(cf)

                        if metal in TOXIC_RESPONSE_FACTORS:
                            er = calculate_ecological_risk_index(conc, bg_val, TOXIC_RESPONSE_FACTORS[metal])
                            ecological_risks.append(er)

                if not contamination_factors:
                    continue

                pli = calculate_pollution_load_index(contamination_factors)
                npi = calculate_nemerow_index(contamination_factors)
                hei = calculate_heavy_metal_evaluation_index(metals, backgrounds)
                total_er = sum(ecological_risks) if ecological_risks else 0

                risk_band, risk_description, risk_percentage = classify_pollution_level(pli, npi, hei)

                # Apply risk filter if specified
                if min_risk and risk_percentage < min_risk:
                    continue

                sample_date = row.get('sample_date', '')
                if pd.isnull(sample_date) or sample_date == '':
                    year = row.get('year', '')
                    month = row.get('month', '')
                    if pd.notnull(year):
                        sample_date = f"{int(year)}-{int(month):02d}" if pd.notnull(month) else f"{int(year)}"

                # Coordinates resolution (support both lower and TitleCase)
                lat_val = row.get('latitude') if ('latitude' in row and pd.notnull(row.get('latitude'))) else row.get('Latitude')
                lon_val = row.get('longitude') if ('longitude' in row and pd.notnull(row.get('longitude'))) else row.get('Longitude')

                result = {
                    "state": str(row.get("state", "") or row.get("state_ut", "") or ""),
                    "district": str(row.get('district', '') or ''),
                    "location": str(row.get("location", "") or ""),
                    "date": str(sample_date) if sample_date else 'N/A',
                    "coordinates": {
                        "lat": safe_float(lat_val),
                        "lon": safe_float(lon_val)
                    },
                    "indices": {
                        "PLI": round(pli, 3),
                        "NPI": round(npi, 3),
                        "HEI": round(hei, 3),
                        "TotalER": round(total_er, 3)
                    },
                    "risk": {
                        "percentage": round(risk_percentage, 2),
                        "band": risk_band,
                        "level": risk_description
                    },
                    "metals": {k: round(v, 3) for k, v in metals.items()},
                    "contamination_factors": {
                        metal: round(calculate_contamination_factor(metals[metal], (backgrounds.get(metal, BACKGROUND_VALUES[metal]))), 3)
                        for metal in metals if metal in BACKGROUND_VALUES
                    }
                }

                all_results.append(result)

            del df
            gc.collect()

        all_results.sort(key=lambda x: x["risk"]["percentage"], reverse=True)

        # Generate map
        map_url_path = None
        try:
            m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles='OpenStreetMap')

            plot_data = [r for r in all_results[:500]
                        if r['coordinates']['lat'] is not None and r['coordinates']['lon'] is not None]

            for result in plot_data:
                band = result['risk']['band']
                color = 'red' if band == 'red' else ('orange' if band == 'yellow' else 'green')

                indices = result['indices']
                popup_html = f"""
                <div style="font-family: Arial; min-width: 220px;">
                    <h4 style="margin: 0 0 10px 0; color: {color};">{result['risk']['level']}</h4>
                    <table style="width: 100%; font-size: 12px;">
                        <tr><td><b>State:</b></td><td>{result['state']}</td></tr>
                        <tr><td><b>District:</b></td><td>{result['district']}</td></tr>
                        <tr><td><b>Location:</b></td><td>{result['location']}</td></tr>
                        <tr><td><b>Date:</b></td><td>{result['date']}</td></tr>
                        <tr><td colspan="2"><hr style="margin: 5px 0;"></td></tr>
                        <tr><td><b>Risk:</b></td><td>{result['risk']['percentage']}%</td></tr>
                        <tr><td><b>PLI:</b></td><td>{indices['PLI']}</td></tr>
                        <tr><td><b>NPI:</b></td><td>{indices['NPI']}</td></tr>
                        <tr><td><b>HEI:</b></td><td>{indices['HEI']}</td></tr>
                    </table>
                </div>
                """

                folium.CircleMarker(
                    location=[result['coordinates']['lat'], result['coordinates']['lon']],
                    radius=7,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.6,
                    weight=2,
                    popup=folium.Popup(popup_html, max_width=320)
                ).add_to(m)

            # Render Folium map to HTML string and cache it for /api/map
            try:
                _last_map_html = m.get_root().render()
            except Exception as e:
                logger.error(f"Map render-to-string error: {e}")
                _last_map_html = None

            # Optional: write to ephemeral /tmp (commonly writeable on PaaS)
            try:
                tmp_dir = os.path.join('/tmp', 'metalsense_static')
                os.makedirs(tmp_dir, exist_ok=True)
                m.save(os.path.join(tmp_dir, 'india_metal_pollution_map.html'))
            except Exception as e:
                logger.warning(f"Map write to /tmp failed: {e}")

            # Always serve via API to avoid filesystem constraints
            map_url_path = '/api/map'

        except Exception as e:
            logger.error(f"Map generation error: {e}")

        # Calculate statistics
        stats: Dict[str, Any] = {}
        if all_results:
            stats = {
                "total_samples": len(all_results),
                "severe_pollution_sites": len([r for r in all_results if r['risk']['band'] == 'red']),
                "moderate_pollution_sites": len([r for r in all_results if r['risk']['band'] == 'yellow']),
                "low_pollution_sites": len([r for r in all_results if r['risk']['band'] == 'green']),
                "average_indices": {
                    "PLI": round(np.mean([r['indices']['PLI'] for r in all_results]), 3),
                    "NPI": round(np.mean([r['indices']['NPI'] for r in all_results]), 3),
                    "HEI": round(np.mean([r['indices']['HEI'] for r in all_results]), 3)
                },
                "average_risk_percentage": round(np.mean([r['risk']['percentage'] for r in all_results]), 2)
            }

        # Generate chart data
        charts = generate_chart_data(all_results)

        gc.collect()

        logger.info(f"Analysis completed with {len(all_results)} samples")
        return {
            "success": True,
            "data": all_results,
            "mapPath": map_url_path,
            "statistics": stats,
            "charts": charts,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in compute_payload: {e}", exc_info=True)
        return {
            "success": False,
            "data": [],
            "mapPath": None,
            "statistics": {},
            "charts": {},
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Helper functions for area-wise aggregation and plotting

def aggregate_indices_by_area(results: List[Dict[str, Any]], by: str = 'state', index: str = 'PLI') -> List[Tuple[str, float]]:
    """Aggregate index values by area key and return sorted averages desc.
    by: one of 'state', 'district', 'location', 'date'
    index: one of 'PLI', 'NPI', 'HEI', 'TotalER'
    """
    valid_by = by if by in ('state', 'district', 'location', 'date') else 'state'
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for r in results:
        key = str(r.get(valid_by, '') or 'Unknown')
        value = r.get('indices', {}).get(index)
        if value is None:
            continue
        try:
            v = float(value)
        except Exception:
            continue
        sums[key] = sums.get(key, 0.0) + v
        counts[key] = counts.get(key, 0) + 1

    averages: List[Tuple[str, float]] = [(k, (sums[k] / counts[k])) for k in counts]
    # Sort by descending average
    averages.sort(key=lambda x: x[1], reverse=True)
    return averages


def render_indices_plot(agg_data: List[Tuple[str, float]], chart_type: str = 'bar', index: str = 'PLI', by: str = 'state', top: int = 10) -> bytes:
    """Render a chart (bar/line) from aggregated data and return PNG bytes."""
    labels = [k for k, _ in agg_data]
    values = [v for _, v in agg_data]

    # Reasonable width based on number of labels
    fig_w = max(8, min(20, 0.6 * max(1, len(labels))))
    fig_h = 4.8
    plt.figure(figsize=(fig_w, fig_h), dpi=150)

    if chart_type not in ('bar', 'line'):
        chart_type = 'bar'

    if chart_type == 'bar':
        bars = plt.bar(labels, values, color='#1f77b4')
        # Add value labels on bars for readability when small
        if len(values) <= 20:
            for bar, val in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}",
                         ha='center', va='bottom', fontsize=8)
    else:
        plt.plot(labels, values, marker='o', linestyle='-', color='#1f77b4')
        if len(values) <= 50:
            for x, y in zip(labels, values):
                plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8)

    plt.title(f"Average {index} by {by.capitalize()} (Top {top})")
    plt.ylabel(index)
    plt.xlabel(by.capitalize())
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.getvalue()


# Flask application
app = Flask(__name__)

# Configure CORS for your website
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://localhost:5173", "https://yourdomain.com"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})


@app.route('/static/<path:filename>')
def static_files(filename):
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    return send_from_directory(static_dir, filename)

@app.route('/api/map', methods=['GET'])
def get_map():
    """Serve the last generated Folium map as HTML"""
    try:
        global _last_map_html
        if _last_map_html:
            resp = make_response(_last_map_html)
            resp.headers['Content-Type'] = 'text/html; charset=utf-8'
            # Intentionally avoid setting X-Frame-Options to allow embedding
            return resp
        else:
            return ("<html><body><p>Map not generated yet. Please run /api/analyze first.</p></body></html>", 404)
    except Exception as e:
        logger.error(f"/api/map error: {e}")
        return ("<html><body><p>Map error</p></body></html>", 500)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "Heavy Metal Pollution Analysis API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/analyze', methods=['GET'])
def analyze():
    """
    Main analysis endpoint with optional filters
    Query parameters:
    - limit: Maximum number of records to analyze (default: 10000)
    - state: Filter by state name (matches either `state` or `state_ut`)
    - district: Filter by district name
    - min_risk: Minimum risk percentage to include (0-100)
    """
    try:
        logger.info("Received analysis request")

        # Get query parameters
        limit = request.args.get('limit', type=int)
        state_filter = request.args.get('state', type=str)
        district_filter = request.args.get('district', type=str)
        min_risk = request.args.get('min_risk', type=float)

        payload = compute_payload(
            limit=limit,
            state_filter=state_filter,
            district_filter=district_filter,
            min_risk=min_risk
        )

        logger.info("Sending analysis response")
        return jsonify(payload)

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "data": [],
            "mapPath": None,
            "statistics": {},
            "charts": {},
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/api/top-polluted', methods=['GET'])
def top_polluted():
    """Get top N most polluted sites"""
    try:
        limit = request.args.get('limit', default=10, type=int)
        payload = compute_payload(limit=10000)  # Get all data first

        if payload['success'] and payload['data']:
            top_sites = payload['data'][:limit]
            return jsonify({
                "success": True,
                "data": top_sites,
                "count": len(top_sites)
            })
        else:
            return jsonify(payload)

    except Exception as e:
        logger.error(f"Top polluted error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/state/<state_name>', methods=['GET'])
def state_analysis(state_name):
    """Get analysis for a specific state"""
    try:
        payload = compute_payload(state_filter=state_name)
        return jsonify(payload)
    except Exception as e:
        logger.error(f"State analysis error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/indices-info', methods=['GET'])
def indices_info():
    """Get information about pollution indices"""
    info = {
        "indices": {
            "PLI": {
                "name": "Pollution Load Index",
                "description": "Geometric mean of contamination factors for all metals",
                "formula": "PLI = (CF₁ × CF₂ × ... × CFₙ)^(1/n)",
                "interpretation": {
                    "< 1": "No pollution",
                    "1 - 2": "Moderate pollution",
                    "2 - 3": "Heavy pollution",
                    "> 3": "Extreme pollution"
                }
            },
            "NPI": {
                "name": "Nemerow Pollution Index",
                "description": "Comprehensive index considering average and maximum contamination",
                "formula": "NPI = √[(CF_avg² + CF_max²) / 2]",
                "interpretation": {
                    "< 0.7": "Clean",
                    "0.7 - 1": "Warning limit",
                    "1 - 2": "Light pollution",
                    "2 - 3": "Moderate pollution",
                    "> 3": "Heavy pollution"
                }
            },
            "HEI": {
                "name": "Heavy Metal Evaluation Index",
                "description": "Sum of ratios of metal concentrations to background values",
                "formula": "HEI = Σ(Cᵢ / Bᵢ)",
                "interpretation": {
                    "< 10": "Low pollution",
                    "10 - 20": "Moderate pollution",
                    "> 20": "High pollution"
                }
            },
            "ER": {
                "name": "Ecological Risk Index",
                "description": "Potential ecological risk considering metal toxicity",
                "formula": "ER = Σ(Tᵢ × CFᵢ)",
                "interpretation": {
                    "< 40": "Low risk",
                    "40 - 80": "Moderate risk",
                    "80 - 160": "Considerable risk",
                    "160 - 320": "High risk",
                    "> 320": "Very high risk"
                }
            }
        },
        "metals": {
            metal: {
                "name": metal,
                "background_value": BACKGROUND_VALUES[metal],
                "toxic_response_factor": TOXIC_RESPONSE_FACTORS.get(metal, "N/A"),
                "unit": "mg/kg"
            }
            for metal in BACKGROUND_VALUES.keys()
        }
    }
    return jsonify(info)


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get only statistics without full data"""
    try:
        payload = compute_payload()
        return jsonify({
            "success": payload['success'],
            "statistics": payload['statistics'],
            "charts": payload['charts'],
            "timestamp": payload['timestamp']
        })
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


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

    logger.info(f"Starting Heavy Metal Pollution Analysis API on {host}:{port}")
    logger.info("Available endpoints:")
    logger.info("  GET /api/analyze - Full pollution analysis")
    logger.info("  GET /api/top-polluted?limit=10 - Top polluted sites")
    logger.info("  GET /api/state/<state_name> - State-specific analysis")
    logger.info("  GET /api/statistics - Statistics only")
    logger.info("  GET /api/indices-info - Index information")
    logger.info("  GET /health - Health check")

    app.run(host=host, port=port, debug=debug, threaded=True)
