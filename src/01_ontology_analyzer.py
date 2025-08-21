"""
Enhanced ESG Ontology Analyzer
Extracts comprehensive ESG indicators and concepts from the ontology for thesis research.
"""

import json
import csv
from datetime import datetime
from rdflib import Graph, Namespace, RDF, RDFS, OWL
from rdflib.plugins.sparql import prepareQuery

class EnhancedESGOntologyAnalyzer:
    def __init__(self, ontology_path):
        self.ontology_path = ontology_path
        self.graph = Graph()
        
        # Define namespaces
        self.ESG = Namespace("http://www.annasvijaya.com/ESGOnt/esgontology#")
        self.RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        self.RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
        self.OWL = Namespace("http://www.w3.org/2002/07/owl#")
        
        # Bind namespaces
        self.graph.bind("esg", self.ESG)
        self.graph.bind("rdf", self.RDF)
        self.graph.bind("rdfs", self.RDFS)
        self.graph.bind("owl", self.OWL)
        
    def load_ontology(self):
        """Load the ESG ontology from OWL file"""
        try:
            self.graph.parse(self.ontology_path, format="xml")
            print(f"Successfully loaded ontology with {len(self.graph)} triples")
            return True
        except Exception as e:
            print(f"Error loading ontology: {e}")
            return False
    
    def extract_performance_indicators(self):
        """Extract explicit PerformanceIndicator instances"""
        query = f"""
        SELECT ?indicator ?label ?comment WHERE {{
            ?indicator rdf:type <{self.ESG}PerformanceIndicator> .
            OPTIONAL {{ ?indicator rdfs:label ?label }}
            OPTIONAL {{ ?indicator rdfs:comment ?comment }}
        }}
        """
        
        results = self.graph.query(query)
        indicators = []
        
        for row in results:
            indicator_name = str(row.indicator).split('#')[-1] if row.indicator else "Unknown"
            indicators.append({
                'name': indicator_name,
                'label': str(row.label) if row.label else indicator_name,
                'comment': str(row.comment) if row.comment else "",
                'type': 'PerformanceIndicator',
                'uri': str(row.indicator)
            })
        
        return indicators
    
    def extract_maturity_indicators(self):
        """Extract explicit MaturityIndicator instances"""
        query = f"""
        SELECT ?indicator ?label ?comment WHERE {{
            ?indicator rdf:type <{self.ESG}MaturityIndicator> .
            OPTIONAL {{ ?indicator rdfs:label ?label }}
            OPTIONAL {{ ?indicator rdfs:comment ?comment }}
        }}
        """
        
        results = self.graph.query(query)
        indicators = []
        
        for row in results:
            indicator_name = str(row.indicator).split('#')[-1] if row.indicator else "Unknown"
            indicators.append({
                'name': indicator_name,
                'label': str(row.label) if row.label else indicator_name,
                'comment': str(row.comment) if row.comment else "",
                'type': 'MaturityIndicator',
                'uri': str(row.indicator)
            })
        
        return indicators
    
    def extract_esg_categories(self):
        """Extract ESG category classes that could represent measurable concepts"""
        # Define ESG-related classes that could be indicators
        esg_classes = [
            'Energy', 'Waste', 'Water', 'GHG_Emissions', 'Biodiversity',
            'Health_and_Safety', 'Employee_Development', 'Employee_Turnover',
            'Board_Diversity', 'Human_Rights', 'Corruption', 'Environmental_Fines',
            'Green_Buildings', 'Green_Products', 'Hazardous_Waste', 'Non-GHG_Air_Emissions',
            'Ozone-Depleting_Gases', 'Product_Safety', 'Resource_Efficiency',
            'WaterEfficiency', 'WasteReduction', 'WasteRecycling', 'Climate_Risk_Mgmt.',
            'Child_Labor', 'Collective_Bargaining', 'Community_and_Society',
            'Corporate_Governance', 'Customer_Relationship', 'Diversity',
            'ESG_Incentives', 'Environmental_Mgmt._System', 'Environmental_Policy',
            'Financial_Inclusion', 'Labor_Practices', 'Privacy_and_IT',
            'Remuneration', 'Supply_Chain', 'Sustainable_Finance', 'Taxes'
        ]
        
        categories = []
        for class_name in esg_classes:
            query = f"""
            SELECT ?cls ?label ?comment WHERE {{
                ?cls rdf:type owl:Class .
                FILTER(CONTAINS(STR(?cls), "{class_name}"))
                OPTIONAL {{ ?cls rdfs:label ?label }}
                OPTIONAL {{ ?cls rdfs:comment ?comment }}
            }}
            """
            
            results = self.graph.query(query)
            for row in results:
                category_name = str(row.cls).split('#')[-1] if row.cls else "Unknown"
                categories.append({
                    'name': category_name,
                    'label': str(row.label) if row.label else category_name,
                    'comment': str(row.comment) if row.comment else "",
                    'type': 'ESGCategory',
                    'uri': str(row.cls)
                })
        
        return categories
    
    def extract_relationships(self):
        """Extract key relationships between concepts"""
        query = """
        SELECT ?subject ?predicate ?object WHERE {
            ?subject ?predicate ?object .
            FILTER(
                CONTAINS(STR(?predicate), "belongsToCategory") ||
                CONTAINS(STR(?predicate), "influencesIndicator") ||
                CONTAINS(STR(?predicate), "impacts") ||
                CONTAINS(STR(?predicate), "hasTarget")
            )
        }
        """
        
        results = self.graph.query(query)
        relationships = []
        
        for row in results:
            relationships.append({
                'subject': str(row.subject).split('#')[-1] if row.subject else "Unknown",
                'predicate': str(row.predicate).split('#')[-1] if row.predicate else "Unknown",
                'object': str(row.object).split('#')[-1] if row.object else "Unknown"
            })
        
        return relationships
    
    def extract_sdg_mappings(self):
        """Extract SDG mappings from the ontology"""
        query = """
        SELECT ?concept ?sdg WHERE {
            ?concept ?predicate ?sdg .
            FILTER(
                CONTAINS(STR(?predicate), "impacts") ||
                CONTAINS(STR(?predicate), "hasTarget")
            )
            FILTER(CONTAINS(STR(?sdg), "SDG"))
        }
        """
        
        results = self.graph.query(query)
        mappings = []
        
        for row in results:
            mappings.append({
                'concept': str(row.concept).split('#')[-1] if row.concept else "Unknown",
                'sdg': str(row.sdg).split('#')[-1] if row.sdg else "Unknown"
            })
        
        return mappings
    
    def analyze_ontology(self):
        """Perform comprehensive analysis of the ESG ontology"""
        if not self.load_ontology():
            return None
        
        # Extract different types of indicators and concepts
        performance_indicators = self.extract_performance_indicators()
        maturity_indicators = self.extract_maturity_indicators()
        esg_categories = self.extract_esg_categories()
        relationships = self.extract_relationships()
        sdg_mappings = self.extract_sdg_mappings()
        
        # Combine all potential indicators
        all_indicators = performance_indicators + maturity_indicators + esg_categories
        
        analysis_result = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'ontology_path': self.ontology_path,
                'total_triples': len(self.graph),
                'total_indicators': len(all_indicators)
            },
            'performance_indicators': performance_indicators,
            'maturity_indicators': maturity_indicators,
            'esg_categories': esg_categories,
            'relationships': relationships,
            'sdg_mappings': sdg_mappings,
            'summary': {
                'performance_indicators_count': len(performance_indicators),
                'maturity_indicators_count': len(maturity_indicators),
                'esg_categories_count': len(esg_categories),
                'total_potential_indicators': len(all_indicators)
            }
        }
        
        return analysis_result
    
    def save_analysis(self, analysis_result, output_path):
        """Save analysis results to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        print(f"Analysis saved to {output_path}")
    
    def save_indicators_mapping(self, analysis_result, csv_path):
        """Save indicator mapping to CSV file"""
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Label', 'Type', 'Comment', 'URI', 'Category'])
            
            # Write all indicators
            for indicator_type in ['performance_indicators', 'maturity_indicators', 'esg_categories']:
                for indicator in analysis_result.get(indicator_type, []):
                    writer.writerow([
                        indicator.get('name', ''),
                        indicator.get('label', ''),
                        indicator.get('type', ''),
                        indicator.get('comment', ''),
                        indicator.get('uri', ''),
                        self._categorize_indicator(indicator.get('name', ''))
                    ])
        
        print(f"Indicator mapping saved to: {csv_path}")
    
    def _categorize_indicator(self, name):
        """Categorize indicator based on name patterns"""
        name_lower = name.lower()
        
        if any(term in name_lower for term in ['energy', 'ghg', 'emission', 'carbon', 'climate']):
            return 'Environmental - Climate & Energy'
        elif any(term in name_lower for term in ['water', 'waste', 'biodiversity', 'forest']):
            return 'Environmental - Resources'
        elif any(term in name_lower for term in ['employee', 'safety', 'health', 'labor', 'human']):
            return 'Social - Workforce'
        elif any(term in name_lower for term in ['community', 'society', 'customer', 'product']):
            return 'Social - Community'
        elif any(term in name_lower for term in ['board', 'governance', 'corruption', 'ethics']):
            return 'Governance - Leadership'
        elif any(term in name_lower for term in ['supply', 'finance', 'tax', 'remuneration']):
            return 'Governance - Operations'
        else:
            return 'Other'

def main():
    # Initialize analyzer
    ontology_path = "../esgontology.owl"
    analyzer = EnhancedESGOntologyAnalyzer(ontology_path)
    
    # Perform analysis
    analysis_result = analyzer.analyze_ontology()
    
    if analysis_result:
        # Save analysis results
        output_path = "data/indicators/esg_ontology_analysis.json"
        csv_path = "data/indicators/esg_ontology_indicators.csv"
        
        analyzer.save_analysis(analysis_result, output_path)
        analyzer.save_indicators_mapping(analysis_result, csv_path)
        
        # Print summary
        print("\n=== Enhanced ESG Ontology Analysis Summary ===")
        print(f"Total Potential Indicators: {analysis_result['summary']['total_potential_indicators']}")
        print(f"Performance Indicators: {analysis_result['summary']['performance_indicators_count']}")
        print(f"Maturity Indicators: {analysis_result['summary']['maturity_indicators_count']}")
        print(f"ESG Category Concepts: {analysis_result['summary']['esg_categories_count']}")
        print(f"Relationships Found: {len(analysis_result['relationships'])}")
        print(f"SDG Mappings Found: {len(analysis_result['sdg_mappings'])}")
        
        print("\n=== Key Concepts Found ===")
        all_indicators = (analysis_result['performance_indicators'] + 
                         analysis_result['maturity_indicators'] + 
                         analysis_result['esg_categories'])
        
        for indicator in all_indicators[:20]:  # Show first 20
            print(f"- {indicator['name']}: {indicator['label']} ({indicator['type']})")
        
        if len(all_indicators) > 20:
            print(f"... and {len(all_indicators) - 20} more concepts")
    
    else:
        print("Failed to analyze ontology")

if __name__ == "__main__":
    main()