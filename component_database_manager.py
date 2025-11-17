# component_database_manager.py
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class ComponentEntry:
    """Database entry for a component with all metadata"""
    name: str
    formula: str
    category: str
    synonyms: List[str] = field(default_factory=list)
    cas_number: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    common_concentrations: Dict[str, str] = field(default_factory=dict)
    elemental_composition: Optional[Dict[str, float]] = None
    approximate_mw: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    type: str = "defined"  # "defined" or "complex"
    pKa: Optional[float] = None
    
    def matches_search(self, query: str) -> bool:
        """Check if component matches search query"""
        query_lower = query.lower()
        
        # Search in name
        if query_lower in self.name.lower():
            return True
        
        # Search in synonyms
        for synonym in self.synonyms:
            if query_lower in synonym.lower():
                return True
        
        # Search in tags
        for tag in self.tags:
            if query_lower in tag.lower():
                return True
        
        # Search in formula
        if query_lower in self.formula.lower():
            return True
            
        # Search in CAS number
        if self.cas_number and query_lower in self.cas_number:
            return True
            
        return False


class ComponentDatabaseManager:
    """Manages the component database with search and category support"""
    
    def __init__(self, database_path: Optional[Path] = None):
        self.database_path = database_path or Path("component_database.json")
        self.components: Dict[str, ComponentEntry] = {}
        self.categories: Dict[str, Dict] = {}
        self._load_database()
    
    def _load_database(self):
        """Load the component database from JSON file"""
        if not self.database_path.exists():
            logger.warning(f"Database file {self.database_path} not found. Creating empty database.")
            self._create_empty_database()
            return
        
        try:
            with open(self.database_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.categories = data.get('categories', {})
            
            # Parse all components into ComponentEntry objects
            for category_key, category_data in self.categories.items():
                for comp_data in category_data.get('components', []):
                    entry = ComponentEntry(
                        name=comp_data['name'],
                        formula=comp_data['formula'],
                        category=category_key,
                        synonyms=comp_data.get('synonyms', []),
                        cas_number=comp_data.get('cas_number'),
                        tags=comp_data.get('tags', []),
                        common_concentrations=comp_data.get('common_concentrations', {}),
                        elemental_composition=comp_data.get('elemental_composition'),
                        approximate_mw=comp_data.get('approximate_mw'),
                        metadata=comp_data.get('metadata', {}),
                        type=comp_data.get('type', 'defined'),
                        pKa=comp_data.get('pKa')
                    )
                    self.components[entry.name] = entry
            
            logger.info(f"Loaded {len(self.components)} components from database")
            
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            self._create_empty_database()
    
    def _create_empty_database(self):
        """Create an empty database structure"""
        self.categories = {
            "carbon_sources": {
                "display_name": "Carbon Sources",
                "description": "Carbon sources for fermentation",
                "components": []
            },
            "nitrogen_sources": {
                "display_name": "Nitrogen Sources", 
                "description": "Nitrogen sources",
                "components": []
            },
            "salts_minerals": {
                "display_name": "Salts & Minerals",
                "description": "Essential salts and minerals",
                "components": []
            }
        }
        self.components = {}
    
    def get_component(self, name: str) -> Optional[ComponentEntry]:
        """Get a component by exact name"""
        return self.components.get(name)
    
    def search_components(self, query: str, category: Optional[str] = None) -> List[ComponentEntry]:
        """Search components by query string and optional category filter"""
        results = []
        
        for comp in self.components.values():
            # Filter by category if specified
            if category and comp.category != category:
                continue
            
            # Check if matches search query
            if comp.matches_search(query):
                results.append(comp)
        
        return results
    
    def get_components_by_category(self, category: str) -> List[ComponentEntry]:
        """Get all components in a specific category"""
        return [comp for comp in self.components.values() 
                if comp.category == category]
    
    def get_components_by_tag(self, tag: str) -> List[ComponentEntry]:
        """Get all components with a specific tag"""
        return [comp for comp in self.components.values() 
                if tag in comp.tags]
    
    def get_all_tags(self) -> set:
        """Get all unique tags in the database"""
        tags = set()
        for comp in self.components.values():
            tags.update(comp.tags)
        return tags
    
    def get_all_categories(self) -> Dict[str, str]:
        """Get all categories with their display names"""
        return {key: cat.get('display_name', key) 
                for key, cat in self.categories.items()}
