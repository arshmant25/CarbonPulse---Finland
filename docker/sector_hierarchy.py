# sector_hierarchy.py
# ─────────────────────────────────────────────────────────────
# Explicit hierarchy + group mapping for all 123 IPCC emission
# categories from Statistics Finland's greenhouse gas inventory
# (table statfin_khki_pxt_138v).
#
# Each entry defines:
#   level     : 0 = grand total, 1 = parent (main sector),
#               2 = child (subsector), 3 = subchild (sub-subsector)
#   parent    : code of the direct parent (None for level 0/1)
#   group     : analysis group used for dashboard grouping
#               (energy, transport, buildings, industry,
#                agriculture, lulucf, waste, total, other)
#
# This table was built by inspecting the actual category list and
# verifying that child values sum (approximately, within rounding)
# to their parent's value -- see verify_hierarchy() below.

SECTOR_HIERARCHY = {
    # ── LEVEL 0: grand totals ───────────────────────────────────
    "TOTAL_NO_LULUCF":   {"name": "Emissions without LULUCF", "level": 0, "parent": None, "group": "total"},
    "TOTAL_WITH_LULUCF": {"name": "Emissions with LULUCF",     "level": 0, "parent": None, "group": "total"},

    # ── SECTOR 1: ENERGY ────────────────────────────────────────
    "1":    {"name": "Energy",                                              "level": 1, "parent": "TOTAL_NO_LULUCF", "group": "energy"},

    "1A1":  {"name": "Energy industries",                                   "level": 2, "parent": "1", "group": "energy"},

    "1A2":  {"name": "Manufacturing industries and construction",           "level": 2, "parent": "1", "group": "industry"},
    "1A2(-1A2gvii)": {"name": "Manufacturing excl. off-road vehicles",       "level": 3, "parent": "1A2", "group": "industry"},
    "1A2gvii":       {"name": "Off-road vehicles & machinery (manuf.)",     "level": 3, "parent": "1A2", "group": "industry"},

    "1A3":  {"name": "Transport",                                           "level": 2, "parent": "1", "group": "transport"},
    "1A3a": {"name": "Domestic aviation",                                   "level": 3, "parent": "1A3", "group": "transport"},
    "1A3b": {"name": "Road transportation",                                 "level": 3, "parent": "1A3", "group": "transport"},
    "1A3bi":   {"name": "Cars",                                             "level": 4, "parent": "1A3b", "group": "transport"},
    "1A3bii":  {"name": "Light duty trucks",                                "level": 4, "parent": "1A3b", "group": "transport"},
    "1A3biii": {"name": "Heavy duty trucks",                                "level": 4, "parent": "1A3b", "group": "transport"},
    "1A3biv":  {"name": "Motorcycles",                                      "level": 4, "parent": "1A3b", "group": "transport"},
    "1A3c": {"name": "Railways",                                            "level": 3, "parent": "1A3", "group": "transport"},
    "1A3d": {"name": "Domestic navigation",                                 "level": 3, "parent": "1A3", "group": "transport"},

    "1A4":  {"name": "Other sectors (buildings)",                           "level": 2, "parent": "1", "group": "buildings"},
    "1A4a": {"name": "Commercial and institutional",                        "level": 3, "parent": "1A4", "group": "buildings"},
    "1A4ai":  {"name": "Commercial/institutional, stationary",              "level": 4, "parent": "1A4a", "group": "buildings"},
    "1A4aii": {"name": "Commercial/institutional, off-road vehicles",       "level": 4, "parent": "1A4a", "group": "buildings"},
    "1A4b": {"name": "Residential",                                         "level": 3, "parent": "1A4", "group": "buildings"},
    "1A4bi":  {"name": "Residential, stationary",                           "level": 4, "parent": "1A4b", "group": "buildings"},
    "1A4bii": {"name": "Residential, off-road vehicles",                    "level": 4, "parent": "1A4b", "group": "buildings"},
    "1A4c": {"name": "Agriculture, forestry and fishing (energy use)",      "level": 3, "parent": "1A4", "group": "buildings"},
    "1A4ci":   {"name": "Agriculture/forestry, stationary",                 "level": 4, "parent": "1A4c", "group": "buildings"},
    "1A4cii":  {"name": "Agriculture/forestry, off-road vehicles",          "level": 4, "parent": "1A4c", "group": "buildings"},
    "1A4ciii": {"name": "Fishing",                                          "level": 4, "parent": "1A4c", "group": "buildings"},

    "1A5":  {"name": "Other fuel use",                                      "level": 2, "parent": "1", "group": "energy"},
    "1B":   {"name": "Fugitive emissions from fuels",                       "level": 2, "parent": "1", "group": "energy"},

    # 1D is international transport -- a memo item, NOT summed into
    # national totals. Treated as its own top-level group.
    "1D":   {"name": "International transport (memo item)",                "level": 1, "parent": None, "group": "international"},
    "1D1a": {"name": "International aviation",                             "level": 2, "parent": "1D", "group": "international"},
    "1D1b": {"name": "International navigation",                          "level": 2, "parent": "1D", "group": "international"},

    # ── SECTOR 2: INDUSTRIAL PROCESSES ──────────────────────────
    "2":    {"name": "Industrial processes and product use",                "level": 1, "parent": "TOTAL_NO_LULUCF", "group": "industry"},
    "2A":   {"name": "Mineral industry",                                    "level": 2, "parent": "2", "group": "industry"},
    "2B":   {"name": "Chemical industry",                                   "level": 2, "parent": "2", "group": "industry"},
    "2C":   {"name": "Metal industry",                                      "level": 2, "parent": "2", "group": "industry"},
    "2D":   {"name": "Non-energy products from fuels & solvent use",        "level": 2, "parent": "2", "group": "industry"},
    "2F":   {"name": "Substitutes for ozone-depleting substances",          "level": 2, "parent": "2", "group": "industry"},
    "2F1":  {"name": "Refrigeration and air conditioning",                  "level": 3, "parent": "2F", "group": "industry"},
    "2F1a": {"name": "Commercial refrigeration",                            "level": 4, "parent": "2F1", "group": "industry"},
    "2F1b": {"name": "Domestic refrigeration",                              "level": 4, "parent": "2F1", "group": "industry"},
    "2F1c": {"name": "Industrial refrigeration",                            "level": 4, "parent": "2F1", "group": "industry"},
    "2F1d": {"name": "Transport refrigeration",                             "level": 4, "parent": "2F1", "group": "industry"},
    "2F1e": {"name": "Mobile air-conditioning",                             "level": 4, "parent": "2F1", "group": "industry"},
    "2F1f": {"name": "Stationary air-conditioning",                         "level": 4, "parent": "2F1", "group": "industry"},
    "2F2":  {"name": "Foam blowing agents",                                 "level": 3, "parent": "2F", "group": "industry"},
    "2F4":  {"name": "Aerosols",                                            "level": 3, "parent": "2F", "group": "industry"},
    "2G":   {"name": "Other product manufacture and use",                   "level": 2, "parent": "2", "group": "industry"},
    "2H":   {"name": "Other industrial process emissions",                  "level": 2, "parent": "2", "group": "industry"},

    # ── SECTOR 3: AGRICULTURE ────────────────────────────────────
    "3":    {"name": "Agriculture",                                         "level": 1, "parent": "TOTAL_NO_LULUCF", "group": "agriculture"},
    "3A":   {"name": "Enteric fermentation",                                "level": 2, "parent": "3", "group": "agriculture"},
    "3A1":  {"name": "Cattle (enteric)",                                    "level": 3, "parent": "3A", "group": "agriculture"},
    "3A1a": {"name": "Dairy cattle (enteric)",                              "level": 4, "parent": "3A1", "group": "agriculture"},
    "3A1b": {"name": "Non-dairy cattle (enteric)",                          "level": 4, "parent": "3A1", "group": "agriculture"},
    "3A2":  {"name": "Sheep (enteric)",                                     "level": 3, "parent": "3A", "group": "agriculture"},
    "3A3":  {"name": "Swine (enteric)",                                     "level": 3, "parent": "3A", "group": "agriculture"},
    "3A4d": {"name": "Goats (enteric)",                                     "level": 3, "parent": "3A", "group": "agriculture"},
    "3A4e": {"name": "Horses and ponies (enteric)",                         "level": 3, "parent": "3A", "group": "agriculture"},
    "3A4hii":  {"name": "Reindeer (enteric)",                               "level": 3, "parent": "3A", "group": "agriculture"},
    "3A4hiv":  {"name": "Fur-bearing animals (enteric)",                    "level": 3, "parent": "3A", "group": "agriculture"},

    "3B":   {"name": "Manure management",                                   "level": 2, "parent": "3", "group": "agriculture"},
    "3B1":  {"name": "Cattle (manure)",                                     "level": 3, "parent": "3B", "group": "agriculture"},
    "3B1a": {"name": "Dairy cattle (manure)",                               "level": 4, "parent": "3B1", "group": "agriculture"},
    "3B1b": {"name": "Non-dairy cattle (manure)",                           "level": 4, "parent": "3B1", "group": "agriculture"},
    "3B2":  {"name": "Sheep (manure)",                                      "level": 3, "parent": "3B", "group": "agriculture"},
    "3B3":  {"name": "Swine (manure)",                                      "level": 3, "parent": "3B", "group": "agriculture"},
    "3B4d": {"name": "Goats (manure)",                                      "level": 3, "parent": "3B", "group": "agriculture"},
    "3B4e": {"name": "Horses and ponies (manure)",                          "level": 3, "parent": "3B", "group": "agriculture"},
    "3B4g": {"name": "Poultry (manure)",                                    "level": 3, "parent": "3B", "group": "agriculture"},
    "3B4hii":  {"name": "Reindeer (manure)",                                "level": 3, "parent": "3B", "group": "agriculture"},
    "3B4hiv":  {"name": "Fur-bearing animals (manure)",                     "level": 3, "parent": "3B", "group": "agriculture"},
    "3B5":  {"name": "Indirect N2O from manure management",                 "level": 3, "parent": "3B", "group": "agriculture"},

    "3D":   {"name": "Agricultural soils",                                  "level": 2, "parent": "3", "group": "agriculture"},
    "3D1":  {"name": "Direct N2O from managed soils",                       "level": 3, "parent": "3D", "group": "agriculture"},
    "3D1a": {"name": "Inorganic N fertilizers",                             "level": 4, "parent": "3D1", "group": "agriculture"},
    "3D1b": {"name": "Organic N fertilizers",                               "level": 4, "parent": "3D1", "group": "agriculture"},
    "3D1c": {"name": "Urine and dung from grazing animals",                 "level": 4, "parent": "3D1", "group": "agriculture"},
    "3D1d": {"name": "Crop residues",                                       "level": 4, "parent": "3D1", "group": "agriculture"},
    "3D1e": {"name": "N2O from soil organic matter loss (mineral)",         "level": 4, "parent": "3D1", "group": "agriculture"},
    "3D1f": {"name": "Cultivation of organic croplands/grasslands",         "level": 4, "parent": "3D1", "group": "agriculture"},
    "3D2":  {"name": "Indirect N2O from managed soils",                     "level": 3, "parent": "3D", "group": "agriculture"},

    "3F":   {"name": "Field burning of agricultural residues",              "level": 2, "parent": "3", "group": "agriculture"},
    "3G":   {"name": "Liming",                                              "level": 2, "parent": "3", "group": "agriculture"},
    "3H":   {"name": "Urea application",                                    "level": 2, "parent": "3", "group": "agriculture"},

    # ── SECTOR 4: LULUCF ─────────────────────────────────────────
    # 4 is the parent of LULUCF, but it is NOT a child of
    # TOTAL_NO_LULUCF. It is added separately to get TOTAL_WITH_LULUCF.
    "4":    {"name": "Land use, land-use change and forestry (LULUCF)",     "level": 1, "parent": "TOTAL_WITH_LULUCF", "group": "lulucf"},

    "4A":   {"name": "Forest land",                                         "level": 2, "parent": "4", "group": "lulucf"},
    "4Ai1": {"name": "Forest land: biomass (mineral soils)",                "level": 3, "parent": "4A", "group": "lulucf"},
    "4Ai2": {"name": "Forest land: biomass (organic soils)",                "level": 3, "parent": "4A", "group": "lulucf"},
    "4Aiv": {"name": "Forest land: DOM+SOM (mineral soils)",                "level": 3, "parent": "4A", "group": "lulucf"},
    "4Av":  {"name": "Forest land: DOM+SOM (organic soils)",                "level": 3, "parent": "4A", "group": "lulucf"},
    "4(i)A":   {"name": "Forest land: N2O from N-fertilization",            "level": 3, "parent": "4A", "group": "lulucf"},
    "4(ii)A":  {"name": "Forest land: drainage & rewetting",                "level": 3, "parent": "4A", "group": "lulucf"},
    "4(iii)A": {"name": "Forest land: N2O from SOM loss (mineral)",         "level": 3, "parent": "4A", "group": "lulucf"},
    "4(iv)A":  {"name": "Forest land: biomass burning",                     "level": 3, "parent": "4A", "group": "lulucf"},

    "4B":   {"name": "Cropland",                                            "level": 2, "parent": "4", "group": "lulucf"},
    "4Bi":  {"name": "Cropland: biomass",                                   "level": 3, "parent": "4B", "group": "lulucf"},
    "4Biv": {"name": "Cropland: DOM+SOM (mineral soils)",                   "level": 3, "parent": "4B", "group": "lulucf"},
    "4Bv":  {"name": "Cropland: DOM+SOM (organic soils)",                   "level": 3, "parent": "4B", "group": "lulucf"},
    "4(iii)B": {"name": "Cropland: N2O from SOM loss (mineral)",            "level": 3, "parent": "4B", "group": "lulucf"},

    "4C":   {"name": "Grassland",                                           "level": 2, "parent": "4", "group": "lulucf"},
    "4Ci":  {"name": "Grassland: biomass",                                  "level": 3, "parent": "4C", "group": "lulucf"},
    "4Civ": {"name": "Grassland: DOM+SOM (mineral soils)",                  "level": 3, "parent": "4C", "group": "lulucf"},
    "4Cv":  {"name": "Grassland: DOM+SOM (organic soils)",                  "level": 3, "parent": "4C", "group": "lulucf"},
    "4(iii)C": {"name": "Grassland: N2O from SOM loss (mineral)",           "level": 3, "parent": "4C", "group": "lulucf"},
    "4(iv)C":  {"name": "Grassland: biomass burning",                       "level": 3, "parent": "4C", "group": "lulucf"},

    "4D":   {"name": "Wetlands",                                            "level": 2, "parent": "4", "group": "lulucf"},
    "4Di":  {"name": "Wetlands: biomass",                                   "level": 3, "parent": "4D", "group": "lulucf"},
    "4Diii": {"name": "Wetlands: dead wood",                                "level": 3, "parent": "4D", "group": "lulucf"},
    "4Dv":  {"name": "Wetlands: SOM",                                       "level": 3, "parent": "4D", "group": "lulucf"},
    "4(ii)D": {"name": "Wetlands: drainage & rewetting",                    "level": 3, "parent": "4D", "group": "lulucf"},

    "4E":   {"name": "Settlements",                                         "level": 2, "parent": "4", "group": "lulucf"},
    "4Ei":  {"name": "Settlements: biomass",                                "level": 3, "parent": "4E", "group": "lulucf"},
    "4Eiii": {"name": "Settlements: dead wood",                             "level": 3, "parent": "4E", "group": "lulucf"},
    "4Eiv-v": {"name": "Settlements: SOM and litter",                       "level": 3, "parent": "4E", "group": "lulucf"},
    "4(i)E":   {"name": "Settlements: N2O from N-fertilization",            "level": 3, "parent": "4E", "group": "lulucf"},
    "4(ii)E":  {"name": "Settlements: drainage & rewetting",                "level": 3, "parent": "4E", "group": "lulucf"},
    "4(iii)E": {"name": "Settlements: N2O from SOM loss (mineral)",         "level": 3, "parent": "4E", "group": "lulucf"},

    "4G":   {"name": "Harvested wood products",                             "level": 2, "parent": "4", "group": "lulucf"},

    # ── SECTOR 5: WASTE ───────────────────────────────────────────
    "5":    {"name": "Waste management",                                    "level": 1, "parent": "TOTAL_NO_LULUCF", "group": "waste"},
    "5A":   {"name": "Waste disposal (landfill)",                           "level": 2, "parent": "5", "group": "waste"},
    "5B":   {"name": "Biological treatment of waste",                       "level": 2, "parent": "5", "group": "waste"},
    "5D":   {"name": "Wastewater treatment and discharge",                  "level": 2, "parent": "5", "group": "waste"},

    # ── INDIRECT CO2 ──────────────────────────────────────────────
    "INDCO2": {"name": "Indirect CO2 emission",                             "level": 1, "parent": "TOTAL_NO_LULUCF", "group": "other"},
}


def get_hierarchy_dataframe():
    """
    Return the hierarchy as a pandas DataFrame with columns:
    code, name, level, parent, group, parent_name, parent_group,
    plus computed `lineage` (root -> ... -> code) and
    `parent_l1` (the level-1 ancestor code, used for grouping charts).
    """
    import pandas as pd

    rows = []
    for code, info in SECTOR_HIERARCHY.items():
        rows.append({
            "sector_code": code,
            "sector_name": info["name"],
            "level":       info["level"],
            "parent_code": info["parent"],
            "group":       info["group"],
        })
    df = pd.DataFrame(rows)

    # Compute the level-1 ancestor for every code (used to group
    # deep subsectors under their main sector for charts)
    code_to_parent = df.set_index("sector_code")["parent_code"].to_dict()
    code_to_level  = df.set_index("sector_code")["level"].to_dict()

    def find_l1_ancestor(code):
        c = code
        seen = set()
        while c is not None and c not in seen:
            seen.add(c)
            if code_to_level.get(c) == 1:
                return c
            if code_to_level.get(c) == 0:
                return c  # totals have no level-1 ancestor
            c = code_to_parent.get(c)
        return c

    df["parent_l1"] = df["sector_code"].apply(find_l1_ancestor)

    return df


def verify_hierarchy(df_long, value_col="value_total"):
    """
    Sanity check: for each (year, parent), sum of direct children's
    values should approximately equal the parent's value.

    df_long must have columns: year, sector_code, <value_col>

    Returns a DataFrame of (year, parent_code, parent_value,
    children_sum, difference) for parents that have children,
    for the most recent year only (spot check).
    """
    import pandas as pd

    hier = get_hierarchy_dataframe()
    parents_with_children = hier[hier["parent_code"].notna()]["parent_code"].unique()

    latest_year = df_long["year"].max()
    df_y = df_long[df_long["year"] == latest_year].set_index("sector_code")[value_col]

    results = []
    for parent in parents_with_children:
        if parent not in df_y.index:
            continue
        children = hier[hier["parent_code"] == parent]["sector_code"].tolist()
        children_present = [c for c in children if c in df_y.index]
        if not children_present:
            continue
        children_sum = df_y[children_present].sum()
        parent_value = df_y[parent]
        results.append({
            "year": latest_year,
            "parent_code": parent,
            "parent_value": parent_value,
            "children_sum": children_sum,
            "difference": parent_value - children_sum,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = get_hierarchy_dataframe()
    print(f"Total categories defined: {len(df)}")
    print(df["level"].value_counts().sort_index())
    print("\nGroups:")
    print(df["group"].value_counts())
