# component_api.py
from typing import List, Dict, Optional, Any
from dataclasses import asdict
import json
from medium_analyzer import Component, ComplexComponent

class ComponentAPI:
    """API layer for GUI integration"""
    
    def __init__(self, library: 'ComponentLibrary'):
        self.library = library
    
    def get_component_for_display(self, name: str) -> Dict[str, Any]:
        """Get component data formatted for GUI display"""
        comp = self.library.get_component(name)
        if not comp:
            return None
        
        return {
            'name': comp.name,
            'formula': comp.formula,
            'mw': comp.mw,
            'category': comp.metadata.get('category', 'Uncategorized'),
            'tags': comp.metadata.get('tags', []),
            'synonyms': comp.metadata.get('synonyms', []),
            'cas_number': comp.metadata.get('cas_number'),
            'common_concentrations': comp.metadata.get('common_concentrations', {}),
            'draggable': True,  # For drag-and-drop support
            'id': f"comp_{comp.name.replace(' ', '_')}"  # Unique ID for GUI
        }
    
    def search_autocomplete(self, partial: str, limit: int = 10) -> List[Dict]:
        """Autocomplete search for GUI search box"""
        results = self.library.search(partial)[:limit]
        return [
            {
                'label': comp.name,
                'value': comp.name,
                'category': comp.metadata.get('category', 'Uncategorized'),
                'formula': comp.formula
            }
            for comp in results
        ]
    
    def get_category_tree(self) -> Dict[str, List[str]]:
        """Get hierarchical structure for GUI tree view"""
        if self.library.db_manager:
            categories = self.library.db_manager.get_all_categories()
            tree = {}
            for cat_key, cat_name in categories.items():
                components = self.library.get_by_category(cat_key)
                tree[cat_name] = [comp.name for comp in components]
            return tree
        else:
            # Fallback
            return {"All Components": list(self.library.components.keys())}
    
    def validate_custom_component(self, data: Dict) -> Dict[str, Any]:
        """Validate user-entered component data for GUI forms"""
        errors = []
        warnings = []
        
        # Required fields
        if not data.get('name'):
            errors.append("Component name is required")
        
        if not data.get('formula'):
            errors.append("Chemical formula is required")
        
        # Try to parse formula
        if data.get('formula'):
            try:
                temp_comp = Component("test", data['formula'])
                temp_comp.parse_formula()
            except Exception as e:
                warnings.append(f"Formula parsing warning: {str(e)}")
        
        # Check for duplicates
        if data.get('name') and data['name'] in self.library.components:
            warnings.append(f"Component '{data['name']}' already exists and will be overwritten")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
