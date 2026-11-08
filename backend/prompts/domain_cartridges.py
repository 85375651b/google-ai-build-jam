"""The 12 Subject-Specific Knowledge Modules and default worldbuilding subject templates."""

from typing import Dict, List, TypedDict


class DomainCartridge(TypedDict):
    key: str
    name: str
    analytical_lenses: str
    cross_domain_dependencies: str
    book_coverage_note: str


DOMAIN_CARTRIDGES: Dict[str, DomainCartridge] = {
    "linguistics": {
        "key": "linguistics",
        "name": "Linguistics, Onomastics & Constructed Languages",
        "analytical_lenses": (
            "Phonological inventories, phonotactic constraints, morphosyntactic alignment "
            "(nominative-accusative, ergative-absolutive, active-stative), historical sound change "
            "(lenition, umlaut, chain shifts), sociolinguistic registers/diglossia, writing systems, "
            "onomastics (anthroponymy: patronymics, clan/lineage names, teknonyms, occupational bynames, "
            "theophoric names; toponymy: hydrographic/orographic compounds, founder eponyms, phonological "
            "erosion of older compounds), and endonyms vs. exonyms."
        ),
        "cross_domain_dependencies": (
            "Geography (isolation vs. Sprachbund contact, topographic place-naming), Social Stratification "
            "(prestigious/liturgical vs. vernacular registers, aristocratic vs. common naming), Technology "
            "(writing media: clay, papyrus, parchment, print, digital handles), Religion (theophoric and taboo names)."
        ),
        "book_coverage_note": (
            "INTENTIONAL BOOK GAP: Mark Rosenfelder explicitly omits conlanging from 'The Planet Construction Kit' "
            "(deferring to 'The Language Construction Kit'). Prioritize scholarly linguistics, historical etymology, and online research."
        ),
    },
    "political_science": {
        "key": "political_science",
        "name": "Political Science & Governance",
        "analytical_lenses": (
            "Sources of legitimacy (sacral, constitutional, martial, performance), selectorate theory and "
            "franchise boundaries, executive-legislative-judicial balance, bureaucratic vs. patrimonial capacity, "
            "succession mechanics, and center-periphery control."
        ),
        "cross_domain_dependencies": (
            "Economics (tax extraction and fiscal capacity), Military (monopoly or fragmentation of armed force), "
            "Geography (communication latency and natural borders), Religion (institutional clerical authority)."
        ),
        "book_coverage_note": (
            "Covered in 'History' and 'Culture' chapters of 'The Planet Construction Kit'; supplement with "
            "comparative historical governance research when institutional detail is needed."
        ),
    },
    "economics": {
        "key": "economics",
        "name": "Economics & Trade",
        "analytical_lenses": (
            "Subsistence base and agricultural surplus, property rights regimes, medium of exchange "
            "(barter, commodity money, coinage, credit/ledger systems), fiscal extraction (tribute, corvée, "
            "tax farming, tariffs), labor organization, and maritime vs. overland transport cost differentials."
        ),
        "cross_domain_dependencies": (
            "Geography & Climate (resource endowments, navigable waterways, arable land), Technology (metallurgy, "
            "transport, energy), Politics (enforcement of contracts, coinage debasement)."
        ),
        "book_coverage_note": (
            "Covered in 'History', 'Culture', and 'Daily Life' chapters of 'The Planet Construction Kit'."
        ),
    },
    "history": {
        "key": "history",
        "name": "History & Chronology",
        "analytical_lenses": (
            "Material and institutional drivers of historical change, path dependence, demographic shocks "
            "(famine, plague, migration), state formation and fragmentation cycles, and historiography "
            "(who records chronicles and what biases they hold)."
        ),
        "cross_domain_dependencies": (
            "Climate & Ecology, Biology (epidemiology), Technology, Warfare, Religion."
        ),
        "book_coverage_note": "Covered in the 'History' chapter of 'The Planet Construction Kit'.",
    },
    "anthropology_sociology": {
        "key": "anthropology_sociology",
        "name": "Anthropology, Culture & Society",
        "analytical_lenses": (
            "Kinship and descent systems (patrilineal, matrilineal, bilateral, clan structures), marriage and "
            "inheritance norms (primogeniture, partible inheritance, ultimogeniture), social stratification/caste, "
            "honor-shame vs. institutional legal norms, rites of passage, and everyday domestic life."
        ),
        "cross_domain_dependencies": (
            "Ecology & Subsistence, Economics (wealth inheritance), Religion, Architecture (household layout)."
        ),
        "book_coverage_note": (
            "Strongly covered in the 'Culture' and 'Daily Life' chapters of 'The Planet Construction Kit'."
        ),
    },
    "religious_studies": {
        "key": "religious_studies",
        "name": "Religion & Mythology",
        "analytical_lenses": (
            "Cosmological/theological structures (polytheism, monolatry, monotheism, animism, non-theistic "
            "philosophies), orthopraxy (ritual correctness) vs. orthodoxy (doctrinal belief), clerical "
            "organization, monastic/temple economics, syncretism, and schism dynamics."
        ),
        "cross_domain_dependencies": (
            "Politics (legitimacy and church-state rivalry), Economics (temple land/tithes), Philosophy, History."
        ),
        "book_coverage_note": "Strongly covered in the 'Religion' chapter of 'The Planet Construction Kit'.",
    },
    "physics_cosmology": {
        "key": "physics_cosmology",
        "name": "Cosmology, Fundamental Physics & Magic Rules",
        "analytical_lenses": (
            "Planetary parameters (stellar class, orbital period, axial tilt, moons, tides, surface gravity), "
            "fundamental physical laws, and internal rule consistency of speculative/magic systems (energy cost, "
            "scarcity, reproducibility, and second-order societal disruptions)."
        ),
        "cross_domain_dependencies": (
            "Climate & Geography, Biology, Technology & Engineering, Warfare, Economics."
        ),
        "book_coverage_note": (
            "Covered in 'Astronomy and Geology' and 'Magic' chapters of 'The Planet Construction Kit'."
        ),
    },
    "biology_ecology": {
        "key": "biology_ecology",
        "name": "Biology, Evolution & Ecology",
        "analytical_lenses": (
            "Trophic pyramids and energy flow, biome adaptation, evolutionary niches, domestication requirements "
            "(diet, growth rate, social hierarchy, tractable temperament), agricultural staple crops, and "
            "disease ecology."
        ),
        "cross_domain_dependencies": (
            "Physics & Climate, Geography, Economics (draft animals and staples), Warfare (cavalry/logistics)."
        ),
        "book_coverage_note": "Strongly covered in the 'Biology' chapter of 'The Planet Construction Kit'.",
    },
    "geography_geology": {
        "key": "geography_geology",
        "name": "Geography, Geology & Climate",
        "analytical_lenses": (
            "Plate tectonics, mountain building (orogeny), atmospheric circulation cells (Hadley/Ferrel/Polar), "
            "orographic precipitation and rain shadows, ocean gyres, watershed drainage basins, and mineral deposits."
        ),
        "cross_domain_dependencies": (
            "Physics (axial tilt/rotation), Biology (biomes), Economics (trade routes and ports), Politics (natural frontiers)."
        ),
        "book_coverage_note": (
            "Strongly covered in 'Astronomy and Geology' and 'Making Maps' chapters of 'The Planet Construction Kit'."
        ),
    },
    "technology_engineering": {
        "key": "technology_engineering",
        "name": "Technology & Engineering",
        "analytical_lenses": (
            "Energy regimes (human/animal muscle, wind/water mills, biomass, fossil fuels, speculative power), "
            "metallurgy and material science, agricultural productivity, transport and communication speed, and "
            "knowledge preservation prerequisites."
        ),
        "cross_domain_dependencies": (
            "Geology (ore/fuel availability), Economics (capital vs. labor costs), Warfare, Social Organization."
        ),
        "book_coverage_note": "Covered in the 'Technology' chapter of 'The Planet Construction Kit'.",
    },
    "military_warfare": {
        "key": "military_warfare",
        "name": "Warfare & Military Organization",
        "analytical_lenses": (
            "Recruitment models (feudal levies, citizen militias, mercenaries, standing professional armies), "
            "logistics and supply range (forage vs. wagon/maritime supply), command and communication, "
            "combined arms interplay, and fortification vs. siegecraft."
        ),
        "cross_domain_dependencies": (
            "Politics (control over the military), Economics (war finance), Geography (terrain bottlenecks), Technology."
        ),
        "book_coverage_note": "Strongly covered in the 'War' chapter of 'The Planet Construction Kit'.",
    },
    "architecture_material_culture": {
        "key": "architecture_material_culture",
        "name": "Architecture, Demographics & Material Culture",
        "analytical_lenses": (
            "Vernacular building materials (timber, mudbrick, dressed stone, concrete), climate adaptation "
            "(passive cooling, snow load, seismic resilience), urban morphology, food preservation, textiles, "
            "and demographic carrying capacity."
        ),
        "cross_domain_dependencies": (
            "Climate & Geography, Flora/Geology, Social Class, Economics, Warfare (city walls)."
        ),
        "book_coverage_note": "Covered in 'Daily Life' and '3-D Modeling' chapters of 'The Planet Construction Kit'.",
    },
}


# The 15 initial optional categories from Step 1 Section 2.1
DEFAULT_WORLD_SUBJECTS: List[Dict[str, str]] = [
    {
        "name": "Cosmology and Fundamental Physics",
        "slug": "cosmology-physics",
        "domain_key": "physics_cosmology",
        "description": "Planetary parameters, fundamental physical laws, magic rules, and cosmic structure.",
    },
    {
        "name": "Geography and Geology",
        "slug": "geography-geology",
        "domain_key": "geography_geology",
        "description": "Continents, plate tectonics, mountain ranges, rivers, and mineral resources.",
    },
    {
        "name": "Climate and Ecology",
        "slug": "climate-ecology",
        "domain_key": "geography_geology",
        "description": "Atmospheric circulation, rainfall, seasons, biomes, and ecological zones.",
    },
    {
        "name": "Biology and Evolution",
        "slug": "biology-evolution",
        "domain_key": "biology_ecology",
        "description": "Species, evolutionary history, flora, fauna, agriculture, and domestication.",
    },
    {
        "name": "History and Chronology",
        "slug": "history-chronology",
        "domain_key": "history",
        "description": "Major eras, historical turning points, migrations, and calendars.",
    },
    {
        "name": "Languages and Linguistics",
        "slug": "languages-linguistics",
        "domain_key": "linguistics",
        "description": "Constructed languages, phonology, grammar, scripts, and naming conventions.",
    },
    {
        "name": "Cultures and Societies",
        "slug": "cultures-societies",
        "domain_key": "anthropology_sociology",
        "description": "Kinship, social stratification, customs, daily life, and social norms.",
    },
    {
        "name": "Religion and Mythology",
        "slug": "religion-mythology",
        "domain_key": "religious_studies",
        "description": "Belief systems, cosmology myths, rituals, priesthoods, and religious institutions.",
    },
    {
        "name": "Politics and Governance",
        "slug": "politics-governance",
        "domain_key": "political_science",
        "description": "Systems of government, sovereignty, suffrage, law, and political institutions.",
    },
    {
        "name": "Economics and Trade",
        "slug": "economics-trade",
        "domain_key": "economics",
        "description": "Currency, taxation, trade networks, property regimes, and labor systems.",
    },
    {
        "name": "Technology and Engineering",
        "slug": "technology-engineering",
        "domain_key": "technology_engineering",
        "description": "Energy sources, metallurgy, infrastructure, tools, and scientific capabilities.",
    },
    {
        "name": "Warfare and Military Organization",
        "slug": "warfare-military",
        "domain_key": "military_warfare",
        "description": "Military recruitment, logistics, strategy, weaponry, and fortifications.",
    },
    {
        "name": "Philosophy and Ideologies",
        "slug": "philosophy-ideologies",
        "domain_key": "religious_studies",
        "description": "Civic ideologies, ethical frameworks, intellectual movements, and worldviews.",
    },
    {
        "name": "Demographics and Population",
        "slug": "demographics-population",
        "domain_key": "architecture_material_culture",
        "description": "Population distribution, urbanization, life expectancy, and migration patterns.",
    },
    {
        "name": "Architecture and Material Culture",
        "slug": "architecture-material-culture",
        "domain_key": "architecture_material_culture",
        "description": "Building styles, urban layout, clothing, cuisine, and everyday artifacts.",
    },
]


EXPERTISE_PEDAGOGICAL_GUIDANCE: Dict[str, str] = {
    "UNKNOWN": (
        "Use clear, accessible prose. Introduce technical terms naturally when helpful and briefly "
        "clarify specialized terminology inline without being patronizing."
    ),
    "BEGINNER": (
        "Explain foundational mechanisms clearly from first principles using concrete examples. "
        "Avoid unexplained jargon overload; introduce at most 1-2 essential technical terms and define them clearly."
    ),
    "INTERMEDIATE": (
        "Use standard domain terminology naturally. Focus on comparative models, structural trade-offs, "
        "and practical cause-and-effect implications."
    ),
    "ADVANCED": (
        "Skip introductory definitions. Converse at an advanced analytical level using accurate "
        "technical terminology, focusing on nuanced systemic interactions and edge cases."
    ),
    "EXPERT": (
        "Treat the author as a peer specialist in this subject. Use precise technical terminology freely "
        "(e.g., IPA/morphosyntax in linguistics, institutional theory in politics) and never explain basic concepts unless asked."
    ),
}
