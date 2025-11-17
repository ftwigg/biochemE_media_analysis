# medium_analyzer.py
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from collections import defaultdict
import numpy as np
from pathlib import Path

# Molecular weights of elements (g/mol)
ATOMIC_WEIGHTS = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'P': 30.974, 'S': 32.06, 'K': 39.098, 'Na': 22.990,
    'Cl': 35.453, 'Ca': 40.078, 'Mg': 24.305, 'Fe': 55.845,
    'Mn': 54.938, 'Zn': 65.38, 'Cu': 63.546, 'Mo': 95.95,
    'Co': 58.933, 'B': 10.81, 'Si': 28.085, 'Se': 78.971
}

# Registry of common ions (easily extendable)
COMMON_IONS = {
    "NH4": {"charge": +1, "elements": {"N": 1, "H": 4}},
    "SO4": {"charge": -2, "elements": {"S": 1, "O": 4}},
    "PO4": {"charge": -3, "elements": {"P": 1, "O": 4}},
    "NO3": {"charge": -1, "elements": {"N": 1, "O": 3}},
    "NO2": {"charge": -2, "elements": {"N": 1, "O": 2}},
    "CO3": {"charge": -2, "elements": {"C": 1, "O": 3}},
    "HCO3": {"charge": -1, "elements": {"H": 1, "C": 1, "O": 3}},
    "Cl":  {"charge": -1, "elements": {"Cl": 1}},
    "Na":  {"charge": +1, "elements": {"Na": 1}},
    "K":   {"charge": +1, "elements": {"K": 1}},
    "Ca":  {"charge": +2, "elements": {"Ca": 1}},
    "Mg":  {"charge": +2, "elements": {"Mg": 1}},
    "CH3COO": {"charge": -1, "elements": {"C": 2, "H": 3, "O": 2}}
}

@dataclass
class FormulaAnalysis:
    """
    Rich representation of a parsed formula.
    Forward-looking hub for further additions of oxidation states, redox bookkeeping, etc.
    """
    # Total elemental composition (includes hydrates where present)
    elements: Dict[str, float] = field(default_factory=dict)
    # Parenthetical or logical fragments, keyed by a label or subformula
    fragments: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Counts of ions from registry (NH4, SO4, etc.)
    ions: Dict[str, float] = field(default_factory=dict)
    # Hydration state as 'n H2O'
    hydrate_nH2O: float = 0.0
    # Net charge inferred from recognized ions (if any)
    net_charge: Optional[float] = None

@dataclass
class Component:
    """Represents a chemical component in the medium"""
    name: str
    formula: str  # Chemical formula (e.g., "C6H12O6", "(NH4)2SO4", "MgSO4·7H2O")
    mw: Optional[float] = None  # Molecular weight (auto-calculated if not provided)
    hydration: int = 0  # Hydration state (e.g., 7 for MgSO4·7H2O)
    metadata: Dict = field(default_factory=dict)  # Additional info

    def __post_init__(self):
        if self.mw is None:
            self.mw = self.calculate_mw()

    # ---------- Formula parsing and tokenization of inputed formatul (no chemistry-aware intepretation yet) ----------

    def _parse_core_formula(self, formula: str) -> Dict[str, float]:
        """
        Parse a formula WITHOUT hydration dots (no '·') but WITH support
        for parentheses and multipliers, e.g. (NH4)2SO4, Ca3(PO4)2.

        Returns elemental composition.
        """
        # Tokens: element, number, '(', ')'
        token_pattern = r'([A-Z][a-z]?|\d*\.?\d+|[()\[\]\{\}])'
        tokens = [t for t in re.findall(token_pattern, formula) if t]

        elements_stack = [defaultdict(float)]
        i = 0

        def is_number(tok: str) -> bool:
            return bool(re.fullmatch(r'\d*\.?\d+', tok))

        def is_element(tok: str) -> bool:
            return bool(re.fullmatch(r'[A-Z][a-z]?', tok))

        while i < len(tokens):
            tok = tokens[i]

            if tok == '(':
                # Start new group
                elements_stack.append(defaultdict(float))
                i += 1

            elif tok == ')':
                # End group: pop it, then apply multiplier if present
                group = elements_stack.pop()
                i += 1
                mult = 1.0
                if i < len(tokens) and is_number(tokens[i]):
                    mult = float(tokens[i])
                    i += 1
                for elem, count in group.items():
                    elements_stack[-1][elem] += count * mult

            elif is_element(tok):
                elem = tok
                i += 1
                mult = 1.0
                if i < len(tokens) and is_number(tokens[i]):
                    mult = float(tokens[i])
                    i += 1
                elements_stack[-1][elem] += mult

            elif is_number(tok):
                # Normally consumed right after element or ')'; if orphan, ignore
                i += 1

            else:
                # Unexpected token; skip
                i += 1

        return dict(elements_stack[0])

    def _extract_parenthetical_fragments(self, formula: str) -> Dict[str, Dict[str, float]]:
        """
        Extract fragments like (NH4)2 or (PO4)3 from the formula.

        Returns mapping:
            subformula (str) -> elemental composition for that subformula
        (before applying multipliers).
        """
        fragments: Dict[str, Dict[str, float]] = {}
        # Non-nested parentheses only (good for most media components).
        pattern = r'\(([^()]*)\)(\d*\.?\d*)'
        for group_formula, multiplier in re.findall(pattern, formula):
            group_formula_stripped = group_formula.strip()
            if group_formula_stripped:
                fragments[group_formula_stripped] = self._parse_core_formula(group_formula_stripped)
        return fragments

    def _expand_for_ions(self, formula: str) -> str:
        """Expand grouped parts of a formula ((), [], {}) with integer
        multipliers into a flat string for ion pattern matching.

        Hydrate dots/interpuncts (·) are stripped since hydrates are handled separately.
        """
        # Work only on the main (non-hydrate) part of the formula
        main = formula.split('·')[0]

        token_pattern = r'([A-Z][a-z]?|\d*\.?\d+|[()\[\]\{\}])'
        tokens = [t for t in re.findall(token_pattern, main) if t]

        def is_number(tok: str) -> bool:
            return bool(re.fullmatch(r'\d*\.?\d+', tok))

        stack = [""]
        i = 0

        while i < len(tokens):
            tok = tokens[i]

            if tok in '([{':
                # Start a new grouped substring
                stack.append("")
                i += 1

            elif tok in ')]}':
                # Close group: pop, apply multiplier if present, and append
                group = stack.pop()
                i += 1
                mult = 1.0
                if i < len(tokens) and is_number(tokens[i]):
                    mult = float(tokens[i])
                    i += 1

                if mult.is_integer():
                    expanded = group * int(mult)
                else:
                    # Non-integer multipliers are unusual; keep symbolic
                    expanded = f"({group}){mult}"

                stack[-1] += expanded

            else:
                # Element symbols and plain numbers just append
                stack[-1] += tok
                i += 1

        return stack[0]

    def _analyze_ions(self, formula: str) -> Dict[str, float]:
        """Count occurrences of known common ions based on COMMON_IONS.

        - Uses _expand_for_ions to handle nested parentheses/brackets.
        - Treats polyatomic ions (NO3, SO4, PO4, etc.) as NOT followed by a digit.
        - Allows monatomic ions (Na, K, Ca, Mg, Cl) to have numeric multipliers (Na2, Ca3, Cl2).
        """
        ion_counts = defaultdict(float)
        if not COMMON_IONS:
            return {}

        expanded = self._expand_for_ions(formula)

        # Split ions into monatomic vs polyatomic based on their element map
        mono_keys = []
        poly_keys = []
        for ion, info in COMMON_IONS.items():
            elems = info.get("elements", {})
            if len(elems) == 1 and next(iter(elems.values())) == 1:
                # Single element, count 1 → monatomic ion (Na+, K+, Ca2+, Cl-, etc.)
                mono_keys.append(ion)
            else:
                # Anything else → polyatomic (NO3-, SO4 2-, HCO3-, CH3COO-, PO4 3-, etc.)
                poly_keys.append(ion)

        # Sort longest-first to reduce ambiguous overlaps
        mono_keys = sorted(mono_keys, key=len, reverse=True)
        poly_keys = sorted(poly_keys, key=len, reverse=True)

        # 1) Polyatomic ions: match only when NOT followed by a digit
        if poly_keys:
            pattern_poly = r'(' + '|'.join(re.escape(ion) for ion in poly_keys) + r')(?!\d)'
            for ion in re.findall(pattern_poly, expanded):
                ion_counts[ion] += 1.0

        # 2) Monatomic ions: allow numeric multipliers (Na2, Ca3, Cl2, etc.)
        if mono_keys:
            pattern_mono = r'(' + '|'.join(re.escape(ion) for ion in mono_keys) + r')(\d*\.?\d*)'
            for ion, count_str in re.findall(pattern_mono, expanded):
                mult = float(count_str) if count_str else 1.0
                ion_counts[ion] += mult

        return dict(ion_counts)

    # ---------- Chemical interpretation of formula after parsing/tokenization ----------

    def analyze_formula(self) -> FormulaAnalysis:
        """
        Rich analysis of the formula:
          - elemental composition (including hydration)
          - fragments for parenthetical groups
          - common ion counts
          - hydration waters
          - inferred net charge from ions (if any)
        """
        # Split on hydration dot
        formula_parts = self.formula.split('·')
        main_formula = formula_parts[0].strip()

        elements = defaultdict(float)

        # 1) Core elemental composition from non-hydrate part
        core_elements = self._parse_core_formula(main_formula)
        for elem, count in core_elements.items():
            elements[elem] += count

        # 2) Hydration: handle single "nH2O" on first dot-part for now
        hydrate_nH2O = 0.0
        if len(formula_parts) > 1:
            hydrate_part = formula_parts[1].strip()
            # e.g. "7H2O" or "H2O"
            water_match = re.fullmatch(r'(\d*\.?\d*)H2O', hydrate_part)
            if water_match:
                n_water = float(water_match.group(1)) if water_match.group(1) else 1.0
                elements["H"] += 2 * n_water
                elements["O"] += n_water
                hydrate_nH2O = n_water
                self.hydration = int(n_water)

        # 3) Parenthetical fragments
        fragments = self._extract_parenthetical_fragments(main_formula)

        # 4) Ion counts and net charge
        ions = self._analyze_ions(main_formula)
        net_charge = None
        if ions:
            charge_sum = 0.0
            for ion, count in ions.items():
                info = COMMON_IONS.get(ion)
                if info and "charge" in info:
                    charge_sum += info["charge"] * count
            net_charge = charge_sum

        return FormulaAnalysis(
            elements=dict(elements),
            fragments=fragments,
            ions=ions,
            hydrate_nH2O=hydrate_nH2O,
            net_charge=net_charge
        )

    # ---------- Simple “element-only” view ----------

    def parse_formula(self) -> Dict[str, float]:
        """
        Elemental composition only.
        Forward-looking stuff is available via analyze_formula().
        """
        return self.analyze_formula().elements

    def calculate_mw(self) -> float:
        """Calculate molecular weight from formula (using elemental composition)."""
        analysis = self.analyze_formula()
        return sum(ATOMIC_WEIGHTS.get(elem, 0) * count
                   for elem, count in analysis.elements.items())

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'formula': self.formula,
            'mw': self.mw,
            'hydration': self.hydration,
            'metadata': self.metadata
        }
        
    def debug_print_analysis(self):
        """Quick helper for development / REPL use."""
        a = self.analyze_formula()
        print(f"Component: {self.name}  ({self.formula})")
        print("  Elements:", a.elements)
        print("  Fragments:", a.fragments)
        print("  Ions:", a.ions)
        print("  Hydrate waters:", a.hydrate_nH2O)
        print("  Net charge:", a.net_charge)

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        return cls(**data)

@dataclass
class ComplexComponent(Component):
    """For complex/undefined components like yeast extract"""
    elemental_composition: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        # For complex components, formula might be descriptive
        if not self.elemental_composition and self.formula:
            super().__post_init__()
        elif self.elemental_composition:
            # Use provided composition
            self.formula = "Complex"
    
    def parse_formula(self) -> Dict[str, float]:
        """Use provided composition or parse formula"""
        if self.elemental_composition:
            return self.elemental_composition.copy()
        return super().parse_formula()

class ComponentLibrary:
    """Manages a library of components with database integration"""
    
    def __init__(self, 
                 library_path: Optional[Path] = None,
                 database_path: Optional[Path] = None,
                 load_from_database: bool = True):
        """
        Initialize component library
        
        Args:
            library_path: Path to user's custom components file
            database_path: Path to the component database
            load_from_database: Whether to load default components from database
        """
        self.library_path = library_path or Path("user_components.json")
        self.components = {}
        
        # Load from database if requested
        if load_from_database:
            try:
                from component_database_manager import ComponentDatabaseManager
                self.db_manager = ComponentDatabaseManager(database_path)
                self._load_from_database()
            except ImportError:
                logger.warning("Database manager not found, loading built-in defaults")
                self.db_manager = None
                self._load_builtin_defaults()
        else:
            self.db_manager = None
            self._load_builtin_defaults()
        
        # Load user's custom components (these override database components)
        if self.library_path.exists():
            self.load_user_components()
    
    def _load_from_database(self):
        """Load components from the database"""
        if not self.db_manager:
            return
        
        for name, entry in self.db_manager.components.items():
            if entry.type == "complex" and entry.elemental_composition:
                comp = ComplexComponent(
                    name=entry.name,
                    formula=entry.formula,
                    elemental_composition=entry.elemental_composition,
                    mw=entry.approximate_mw,
                    metadata={
                        **entry.metadata,
                        'category': entry.category,
                        'tags': entry.tags,
                        'synonyms': entry.synonyms,
                        'cas_number': entry.cas_number
                    }
                )
            else:
                comp = Component(
                    name=entry.name,
                    formula=entry.formula,
                    metadata={
                        'category': entry.category,
                        'tags': entry.tags,
                        'synonyms': entry.synonyms,
                        'cas_number': entry.cas_number,
                        'common_concentrations': entry.common_concentrations,
                        **entry.metadata
                    }
                )
            self.components[name] = comp
        
        logger.info(f"Loaded {len(self.components)} components from database")
    
    def _load_builtin_defaults(self):
        """Fallback: Load minimal built-in defaults if database unavailable"""
        defaults = [
            Component("Glucose", "C6H12O6"),
            Component("Ammonium Sulfate", "(NH4)2SO4"),
            Component("Magnesium Sulfate Heptahydrate", "MgSO4·7H2O"),
            Component("Sodium Chloride", "NaCl"),
        ]
        for comp in defaults:
            self.components[comp.name] = comp
    
    def load_user_components(self):
        """Load user's custom components from file"""
        try:
            with open(self.library_path, 'r') as f:
                data = json.load(f)
            
            for name, comp_data in data.items():
                if comp_data.get('elemental_composition'):
                    comp = ComplexComponent.from_dict(comp_data)
                else:
                    comp = Component.from_dict(comp_data)
                # User components override database components
                self.components[name] = comp
                
            logger.info(f"Loaded {len(data)} user components")
        except Exception as e:
            logger.error(f"Error loading user components: {e}")
    
    def search(self, query: str, category: Optional[str] = None) -> List[Component]:
        """Search components using database search capabilities"""
        if self.db_manager:
            entries = self.db_manager.search_components(query, category)
            return [self.components.get(e.name) for e in entries 
                   if e.name in self.components]
        else:
            # Fallback: simple name search
            results = []
            query_lower = query.lower()
            for name, comp in self.components.items():
                if query_lower in name.lower():
                    results.append(comp)
            return results
    
    def get_by_category(self, category: str) -> List[Component]:
        """Get all components in a category"""
        if self.db_manager:
            entries = self.db_manager.get_components_by_category(category)
            return [self.components.get(e.name) for e in entries 
                   if e.name in self.components]
        else:
            # Fallback: check metadata
            return [comp for comp in self.components.values()
                   if comp.metadata.get('category') == category]
    
    def get_by_tag(self, tag: str) -> List[Component]:
        """Get all components with a specific tag"""
        if self.db_manager:
            entries = self.db_manager.get_components_by_tag(tag)
            return [self.components.get(e.name) for e in entries 
                   if e.name in self.components]
        else:
            # Fallback: check metadata
            return [comp for comp in self.components.values()
                   if tag in comp.metadata.get('tags', [])]

class Recipe:
    """Represents a medium recipe"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.components: List[tuple[Component, float, str]] = []  # (component, concentration, unit)
        self.base_recipes: List[tuple['Recipe', float]] = []  # For recipe mixing
    
    def add_component(self, component: Component, concentration: float, unit: str = "g/L"):
        """Add a component to the recipe"""
        # Convert to g/L if needed
        conc_g_per_l = self._convert_to_g_per_l(concentration, unit, component)
        self.components.append((component, conc_g_per_l, "g/L"))
    
    def add_base_recipe(self, recipe: 'Recipe', dilution_factor: float = 1.0):
        """Add another recipe as a base (for mixing)"""
        self.base_recipes.append((recipe, dilution_factor))
    
    def _convert_to_g_per_l(self, concentration: float, unit: str, component: Component) -> float:
        """Convert various units to g/L"""
        conversions = {
            'g/L': 1.0,
            'mg/L': 0.001,
            'ug/L': 0.000001,
            'mg/mL': 1.0,
            'g/mL': 1000.0,
            'M': component.mw,  # Molar to g/L
            'mM': component.mw * 0.001,
            'uM': component.mw * 0.000001,
        }
        
        if unit not in conversions:
            raise ValueError(f"Unknown unit: {unit}")
        
        return concentration * conversions[unit]
    
    def get_total_composition(self) -> List[tuple[Component, float]]:
        """Get all components including those from base recipes"""
        total = []
        
        # Add components from base recipes
        for base_recipe, dilution in self.base_recipes:
            base_components = base_recipe.get_total_composition()
            for comp, conc in base_components:
                total.append((comp, conc * dilution))
        
        # Add direct components
        for comp, conc, _ in self.components:
            total.append((comp, conc))
        
        # Combine duplicates
        combined = defaultdict(float)
        for comp, conc in total:
            combined[comp.name] += conc
        
        # Get unique components
        comp_dict = {comp.name: comp for comp, _ in total}
        
        return [(comp_dict[name], conc) for name, conc in combined.items()]
    
    def analyze(self) -> 'MediumAnalysis':
        """Analyze the recipe composition"""
        return MediumAnalysis(self)


class MediumAnalysis:
    """Analyzes medium composition"""
    
    def __init__(self, recipe: Recipe):
        self.recipe = recipe
        self.composition = recipe.get_total_composition()
        self._elemental_composition = None
        self._calculate_all()
    
    def _calculate_all(self):
        """Calculate all metrics"""
        self._elemental_composition = self._calculate_elemental_composition()
    
    def _calculate_elemental_composition(self) -> Dict[str, float]:
        """Calculate total elemental composition in g/L"""
        elements = defaultdict(float)
        
        for component, conc_g_L in self.composition:
            comp_elements = component.parse_formula()
            
            for element, count in comp_elements.items():
                # Convert from g/L of compound to g/L of element
                element_mass_fraction = (count * ATOMIC_WEIGHTS.get(element, 0)) / component.mw
                elements[element] += conc_g_L * element_mass_fraction
        
        return dict(elements)
    
    def get_elemental_composition(self, unit: str = "g/L") -> Dict[str, float]:
        """Get elemental composition in specified units"""
        if unit == "g/L":
            return self._elemental_composition.copy()
        elif unit == "mg/L" or unit == "ppm":
            return {e: c * 1000 for e, c in self._elemental_composition.items()}
        elif unit == "mM":
            return {e: (c * 1000) / ATOMIC_WEIGHTS[e] 
                   for e, c in self._elemental_composition.items()}
        else:
            raise ValueError(f"Unknown unit: {unit}")
    
    def get_cn_ratio(self) -> Optional[float]:
        """Calculate C/N ratio (mass basis)"""
        c_mass = self._elemental_composition.get('C', 0)
        n_mass = self._elemental_composition.get('N', 0)
        
        if n_mass > 0:
            return c_mass / n_mass
        return None
    
    def get_molar_cn_ratio(self) -> Optional[float]:
        """Calculate C/N ratio (molar basis)"""
        c_mass = self._elemental_composition.get('C', 0)
        n_mass = self._elemental_composition.get('N', 0)
        
        if n_mass > 0:
            c_moles = c_mass / ATOMIC_WEIGHTS['C']
            n_moles = n_mass / ATOMIC_WEIGHTS['N']
            return c_moles / n_moles
        return None
    
    def get_trace_metals(self) -> Dict[str, float]:
        """Get trace metal concentrations in ppm (mg/L)"""
        trace_metals = ['Fe', 'Mn', 'Zn', 'Cu', 'Mo', 'Co', 'Se']
        return {
            metal: self._elemental_composition.get(metal, 0) * 1000
            for metal in trace_metals
            if metal in self._elemental_composition
        }
    
    def get_summary(self) -> dict:
        """Get comprehensive analysis summary"""
        return {
            'recipe_name': self.recipe.name,
            'total_components': len(self.composition),
            'elemental_composition_g_L': self.get_elemental_composition("g/L"),
            'elemental_composition_ppm': self.get_elemental_composition("ppm"),
            'cn_ratio_mass': self.get_cn_ratio(),
            'cn_ratio_molar': self.get_molar_cn_ratio(),
            'trace_metals_ppm': self.get_trace_metals(),
            'total_mass_g_L': sum(conc for _, conc in self.composition)
        }
    
    def print_report(self):
        """Print a formatted analysis report"""
        print(f"\n{'='*60}")
        print(f"Medium Analysis Report: {self.recipe.name}")
        print(f"{'='*60}")
        
        print("\nComponent Composition:")
        print(f"{'Component':<30} {'Conc (g/L)':>12}")
        print("-" * 43)
        for comp, conc in self.composition:
            print(f"{comp.name:<30} {conc:>12.4f}")
        
        print(f"\nTotal Mass: {sum(c for _, c in self.composition):.4f} g/L")
        
        print("\nElemental Composition:")
        print(f"{'Element':<10} {'g/L':>12} {'ppm':>12}")
        print("-" * 35)
        for element in sorted(self._elemental_composition.keys()):
            g_l = self._elemental_composition[element]
            ppm = g_l * 1000
            print(f"{element:<10} {g_l:>12.4f} {ppm:>12.2f}")
        
        print("\nKey Ratios:")
        cn_mass = self.get_cn_ratio()
        cn_molar = self.get_molar_cn_ratio()
        if cn_mass:
            print(f"  C/N ratio (mass):  {cn_mass:.2f}")
            print(f"  C/N ratio (molar): {cn_molar:.2f}")
        
        trace_metals = self.get_trace_metals()
        if trace_metals:
            print("\nTrace Metals (ppm):")
            for metal, ppm in trace_metals.items():
                print(f"  {metal}: {ppm:.4f}")
