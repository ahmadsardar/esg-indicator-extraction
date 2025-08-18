"""
ESG Ontology Analyzer
Parses the ESG ontology file to extract indicators and their relationships
"""

import rdflib
from rdflib import Graph, Namespace, URIRef
import json
from typing import Dict, List, Set
import pandas as pd

class ESGOntologyAnalyzer:
    def __init__(self, ontology_path: str):
        """Initialize the ontology analyzer"""
        self.graph = Graph()
        self.ontology_path = ontology_path
        
        # Define namespaces
        self.ESG = Namespace("http://www.annasvijaya.com/ESGOnt/esgontology#")
        self.RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        self.RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
        self.OWL = Namespace("http://www.w3.org/2002/07/owl#")
        
        self.indicators = {}
        self.categories = {}
        self.relationships = []
        
    def load_ontology(self):
        """Load the ESG ontology from OWL file"""
        try:
            self.graph.parse(self.ontology_path, format="xml")
            print(f"Successfully loaded ontology with {len(self.graph)} triples")
        except Exception as e:
            print(f"Error loading ontology: {e}")
            
    def extract_performance_indicators(self) -> Dict:
        """Extract all performance indicators from the ontology"""
        indicators = {}
        
        # Query for all performance indicators
        query = f"""
        SELECT ?indicator ?label ?comment
        WHERE {{
            ?indicator rdf:type <{self.ESG}PerformanceIndicator> .
            OPTIONAL {{ ?indicator rdfs:label ?label }}
            OPTIONAL {{ ?indicator rdfs:comment ?comment }}
        }}
        """
        
        results = self.graph.query(query)
        
        for row in results:
            indicator_uri = str(row.indicator)
            indicator_name = indicator_uri.split('#')[-1] if '#' in indicator_uri else indicator_uri
            
            indicators[indicator_name] = {
                'uri': indicator_uri,
                'label': str(row.label) if row.label else indicator_name,
                'comment': str(row.comment) if row.comment else '',
                'type': 'PerformanceIndicator'
            }
            
        return indicators
    
    def extract_maturity_indicators(self) -> Dict:
        """Extract all maturity indicators from the ontology"""
        indicators = {}
        
        query = f"""
        SELECT ?indicator ?label ?comment
        WHERE {{
            ?indicator rdf:type <{self.ESG}MaturityIndicator> .
            OPTIONAL {{ ?indicator rdfs:label ?label }}
            OPTIONAL {{ ?indicator rdfs:comment ?comment }}
        }}
        """
        
        results = self.graph.query(query)
        
        for row in results:
            indicator_uri = str(row.indicator)
            indicator_name = indicator_uri.split('#')[-1] if '#' in indicator_uri else indicator_uri
            
            indicators[indicator_name] = {
                'uri': indicator_uri,
                'label': str(row.label) if row.label else indicator_name,
                'comment': str(row.comment) if row.comment else '',
                'type': 'MaturityIndicator'
            }
            
        return indicators
    
    def extract_categories(self) -> Dict:
        """Extract ESG categories and domains"""
        categories = {}
        
        # Extract categories
        query = f"""
        SELECT ?category ?label ?comment ?domain
        WHERE {{
            ?category rdf:type <{self.ESG}Category> .
            OPTIONAL {{ ?category rdfs:label ?label }}
            OPTIONAL {{ ?category rdfs:comment ?comment }}
            OPTIONAL {{ ?category <{self.ESG}hasDomain> ?domain }}
        }}
        """
        
        results = self.graph.query(query)
        
        for row in results:
            category_uri = str(row.category)
            category_name = category_uri.split('#')[-1] if '#' in category_uri else category_uri
            
            domain_name = ''
            if row.domain:
                domain_uri = str(row.domain)
                domain_name = domain_uri.split('#')[-1] if '#' in domain_uri else domain_uri
            
            categories[category_name] = {
                'uri': category_uri,
                'label': str(row.label) if row.label else category_name,
                'comment': str(row.comment) if row.comment else '',
                'domain': domain_name
            }
            
        return categories
    
    def extract_relationships(self) -> List[Dict]:
        """Extract relationships between indicators and categories"""
        relationships = []
        
        # Query for belongsToCategory relationships
        query = f"""
        SELECT ?indicator ?category
        WHERE {{
            ?indicator <{self.ESG}belongsToCategory> ?category .
        }}
        """
        
        results = self.graph.query(query)
        
        for row in results:
            indicator_name = str(row.indicator).split('#')[-1]
            category_name = str(row.category).split('#')[-1]
            
            relationships.append({
                'indicator': indicator_name,
                'category': category_name,
                'relationship': 'belongsToCategory'
            })
            
        return relationships
    
    def analyze_ontology(self) -> Dict:
        """Perform complete ontology analysis"""
        self.load_ontology()
        
        # Extract all components
        performance_indicators = self.extract_performance_indicators()
        maturity_indicators = self.extract_maturity_indicators()
        categories = self.extract_categories()
        relationships = self.extract_relationships()
        
        # Combine all indicators
        all_indicators = {**performance_indicators, **maturity_indicators}
        
        analysis_result = {
            'indicators': all_indicators,
            'categories': categories,
            'relationships': relationships,
            'summary': {
                'total_indicators': len(all_indicators),
                'performance_indicators': len(performance_indicators),
                'maturity_indicators': len(maturity_indicators),
                'categories': len(categories)
            }
        }
        
        return analysis_result
    
    def save_analysis(self, analysis_result: Dict, output_path: str):
        """Save analysis results to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        print(f"Analysis saved to {output_path}")
    
    def create_indicator_mapping(self, analysis_result: Dict) -> pd.DataFrame:
        """Create a mapping table of indicators for annotation purposes"""
        indicators_data = []
        
        for name, details in analysis_result['indicators'].items():
            indicators_data.append({
                'indicator_name': name,
                'label': details['label'],
                'type': details['type'],
                'comment': details['comment'],
                'category': '',  # To be filled based on relationships
                'domain': ''     # To be filled based on category domain
            })
        
        # Add relationship information
        for rel in analysis_result['relationships']:
            for indicator in indicators_data:
                if indicator['indicator_name'] == rel['indicator']:
                    indicator['category'] = rel['category']
                    # Find domain for this category
                    if rel['category'] in analysis_result['categories']:
                        indicator['domain'] = analysis_result['categories'][rel['category']].get('domain', '')
        
        return pd.DataFrame(indicators_data)

def main():
    """Main function to run the ontology analysis"""
    ontology_path = "../esgontology.owl"
    output_path = "../Project/data/ontology/esg_ontology_analysis.json"
    mapping_path = "../Project/data/ontology/esg_indicators_mapping.csv"
    
    analyzer = ESGOntologyAnalyzer(ontology_path)
    analysis_result = analyzer.analyze_ontology()
    
    # Save analysis
    analyzer.save_analysis(analysis_result, output_path)
    
    # Create and save indicator mapping
    mapping_df = analyzer.create_indicator_mapping(analysis_result)
    mapping_df.to_csv(mapping_path, index=False)
    
    # Print summary
    print("\n=== ESG Ontology Analysis Summary ===")
    print(f"Total Indicators: {analysis_result['summary']['total_indicators']}")
    print(f"Performance Indicators: {analysis_result['summary']['performance_indicators']}")
    print(f"Maturity Indicators: {analysis_result['summary']['maturity_indicators']}")
    print(f"Categories: {analysis_result['summary']['categories']}")
    
    print("\n=== Key Indicators Found ===")
    for name, details in list(analysis_result['indicators'].items())[:10]:
        print(f"- {name}: {details['label']} ({details['type']})")
    
    print(f"\nIndicator mapping saved to: {mapping_path}")

if __name__ == "__main__":
    main()