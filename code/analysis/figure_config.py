"""Display labels, feature categories and colors shared by the figures."""

CATEGORY_PALETTE = {
    "Building": "#CC3311",
    "Amenity": "#009988",
    "Accessibility": "#EE3377",
    "Location": "#0077BB",
    "Time": "#EE7733",
    "Other": "#AAAAAA",
}

HORIZON_COLORS = ["#CC3311", "#009988", "#0077BB", "#EE7733"]

VARIABLE_DISPLAY = {
    "area_numeric": "Area (m²)",
    "floor": "Floor",
    "build_year": "Build Year",
    "building_age": "Building Age",
    "building_age_sq": "Building Age²",
    "area_sq": "Area²",
    "log_area": "Log Area",
    "area_x_floor": "Area × Floor",
    "total_parking": "Total Parking",
    "cctv_count": "CCTV Count",
    "elevator_count": "Elevator Count",
    "edu_count": "Education Count",
    "welfare_count": "Welfare Count",
    "welfare_facility": "Welfare Facility",
    "education_facility": "Education Facility",
    "convenient_facility": "Convenient Facility",
    "total_el_chargers": "Total EV Chargers",
    "has_parking": "Has Parking",
    "has_elevator": "Has Elevator",
    "has_cctv": "Has CCTV",
    "has_el_charger": "Has EV Charger",
    "has_subway": "Has Subway",
    "subway_line": "Subway Line",
    "subway_station": "Subway Station",
    "kaptd_wtimesub": "Subway Commute Time",
    "contract_ym": "Contract Month",
    "time_trend": "Time Trend",
    "month": "Month",
    "month_effect": "Month Effect",
    "year": "Year",
    "location_gu": "District",
    "location_complex": "Apartment complex",
    "location_address": "Address",
    "location_road": "Road",
    "District": "District",
    "kapt_code": "Complex Code",
    "district": "District",
}

STRUCTURAL_FEATURES = {
    "area_numeric",
    "floor",
    "build_year",
    "building_age",
    "area_sq",
    "log_area",
    "building_age_sq",
    "area_x_floor",
}
AMENITY_FEATURES = {
    "total_parking",
    "cctv_count",
    "elevator_count",
    "edu_count",
    "welfare_count",
    "welfare_facility",
    "education_facility",
    "convenient_facility",
    "total_el_chargers",
    "has_el_charger",
    "has_parking",
    "has_elevator",
    "has_cctv",
}
ACCESSIBILITY_FEATURES = {
    "subway_line",
    "subway_station",
    "kaptd_wtimesub",
    "has_subway",
    "subway_within_500m",
    "subway_time",
}
LOCATION_FEATURES = {"district", "시군구", "번지", "단지명", "road_name", "kapt_code"}
TIME_FEATURES = {"contract_ym", "time_trend", "month", "month_effect", "year"}


def categorize_feature(fname):
    if fname in STRUCTURAL_FEATURES:
        return "Building"
    if fname in AMENITY_FEATURES:
        return "Amenity"
    if fname in ACCESSIBILITY_FEATURES:
        return "Accessibility"
    if fname in LOCATION_FEATURES:
        return "Location"
    if fname in TIME_FEATURES:
        return "Time"
    if (
        fname.startswith("gu_")
        or fname.startswith("location_")
        or fname.startswith("area_x_district")
    ):
        return "Location"
    if fname.startswith("month_"):
        return "Time"
    return "Other"


def display_name(fname):
    return VARIABLE_DISPLAY.get(fname, fname.replace("_", " ").title())
