#!/usr/bin/env python3
"""
ESG Indicators Expansion Script

This script expands the ESG indicator set from the original 2 indicators (EnergyEfficiency, WasteReductionIndicator)
to a comprehensive set based on major ESG frameworks: GRI, SASB, TCFD, and industry best practices.

Sources:
- Global Reporting Initiative (GRI) Standards
- Sustainability Accounting Standards Board (SASB) Standards  
- Task Force on Climate-related Financial Disclosures (TCFD)
- EU Taxonomy for Sustainable Activities
- UN Sustainable Development Goals (SDGs)

Author: Thesis Project
Date: 2024
"""

import json
import csv
from typing import Dict, List, Any
from datetime import datetime

class ESGIndicatorExpansion:
    def __init__(self):
        self.indicators = []
        self.sources = {
            "GRI": "Global Reporting Initiative Standards - https://www.globalreporting.org/standards",
            "SASB": "Sustainability Accounting Standards Board - https://sasb.ifrs.org/standards/",
            "TCFD": "Task Force on Climate-related Financial Disclosures",
            "EU_TAXONOMY": "EU Taxonomy for Sustainable Activities",
            "SDG": "UN Sustainable Development Goals",
            "ORIGINAL": "Original esgontology.owl file"
        }
        
    def add_indicator(self, name: str, category: str, subcategory: str, description: str, 
                    source: str, metric_type: str = "quantitative", unit: str = None):
        """Add an ESG indicator with full documentation"""
        indicator = {
            "name": name,
            "category": category,  # Environmental, Social, Governance
            "subcategory": subcategory,
            "description": description,
            "source": source,
            "source_reference": self.sources.get(source, "Unknown"),
            "metric_type": metric_type,
            "unit": unit,
            "added_date": datetime.now().isoformat()
        }
        self.indicators.append(indicator)
        
    def load_original_indicators(self):
        """Load the original 2 indicators from ontology analysis"""
        self.add_indicator(
            "EnergyEfficiency", "Environmental", "Energy Management",
            "Measures related to energy efficiency improvements and energy consumption optimization",
            "ORIGINAL", "quantitative", "kWh/unit or %"
        )
        
        self.add_indicator(
            "WasteReductionIndicator", "Environmental", "Waste Management", 
            "Indicators measuring waste reduction efforts and circular economy practices",
            "ORIGINAL", "quantitative", "tons or %"
        )
        
    def add_environmental_indicators(self):
        """Add comprehensive environmental indicators from GRI, SASB, TCFD"""
        
        # Climate Change & GHG Emissions (GRI 305, SASB, TCFD)
        self.add_indicator(
            "GHGEmissionsScope1", "Environmental", "Climate Change",
            "Direct greenhouse gas emissions from owned or controlled sources",
            "GRI", "quantitative", "tCO2e"
        )
        
        self.add_indicator(
            "GHGEmissionsScope2", "Environmental", "Climate Change",
            "Indirect greenhouse gas emissions from purchased energy",
            "GRI", "quantitative", "tCO2e"
        )
        
        self.add_indicator(
            "GHGEmissionsScope3", "Environmental", "Climate Change",
            "Other indirect greenhouse gas emissions in value chain",
            "GRI", "quantitative", "tCO2e"
        )
        
        self.add_indicator(
            "CarbonIntensity", "Environmental", "Climate Change",
            "Carbon emissions per unit of revenue or production",
            "SASB", "quantitative", "tCO2e/revenue"
        )
        
        # Energy Management (GRI 302)
        self.add_indicator(
            "EnergyConsumption", "Environmental", "Energy Management",
            "Total energy consumption within the organization",
            "GRI", "quantitative", "GJ or MWh"
        )
        
        self.add_indicator(
            "RenewableEnergyPercentage", "Environmental", "Energy Management",
            "Percentage of energy consumption from renewable sources",
            "GRI", "quantitative", "%"
        )
        
        self.add_indicator(
            "EnergyIntensity", "Environmental", "Energy Management",
            "Energy consumption per unit of activity or revenue",
            "GRI", "quantitative", "GJ/revenue"
        )
        
        # Water Management (GRI 303)
        self.add_indicator(
            "WaterConsumption", "Environmental", "Water Management",
            "Total water consumption by the organization",
            "GRI", "quantitative", "m³"
        )
        
        self.add_indicator(
            "WaterRecycling", "Environmental", "Water Management",
            "Percentage of water recycled and reused",
            "GRI", "quantitative", "%"
        )
        
        self.add_indicator(
            "WaterStressAreas", "Environmental", "Water Management",
            "Water consumption in water-stressed areas",
            "SASB", "quantitative", "m³"
        )
        
        # Waste Management (GRI 306)
        self.add_indicator(
            "WasteGenerated", "Environmental", "Waste Management",
            "Total weight of waste generated",
            "GRI", "quantitative", "tons"
        )
        
        self.add_indicator(
            "WasteRecycled", "Environmental", "Waste Management",
            "Percentage of waste diverted from disposal through recycling",
            "GRI", "quantitative", "%"
        )
        
        self.add_indicator(
            "HazardousWaste", "Environmental", "Waste Management",
            "Total weight of hazardous waste generated",
            "GRI", "quantitative", "tons"
        )
        
        # Biodiversity (GRI 304)
        self.add_indicator(
            "BiodiversityImpact", "Environmental", "Biodiversity",
            "Significant impacts on biodiversity in protected areas",
            "GRI", "qualitative", "description"
        )
        
        self.add_indicator(
            "LandUse", "Environmental", "Biodiversity",
            "Total land area owned, leased, or managed for production activities",
            "SASB", "quantitative", "hectares"
        )
        
        # Air Quality (SASB)
        self.add_indicator(
            "AirEmissions", "Environmental", "Air Quality",
            "Air emissions of pollutants (NOx, SOx, particulates)",
            "SASB", "quantitative", "tons"
        )
        
    def add_social_indicators(self):
        """Add comprehensive social indicators from GRI and SASB"""
        
        # Employment (GRI 401)
        self.add_indicator(
            "EmployeeTurnover", "Social", "Employment",
            "Total number and rates of new employee hires and employee turnover",
            "GRI", "quantitative", "rate %"
        )
        
        self.add_indicator(
            "EmployeeBenefits", "Social", "Employment",
            "Benefits provided to full-time employees",
            "GRI", "qualitative", "description"
        )
        
        # Diversity & Inclusion (GRI 405)
        self.add_indicator(
            "BoardDiversity", "Social", "Diversity & Inclusion",
            "Diversity of governance bodies and employees by gender, age, minority groups",
            "GRI", "quantitative", "%"
        )
        
        self.add_indicator(
            "GenderPayGap", "Social", "Diversity & Inclusion",
            "Ratio of basic salary and remuneration of women to men",
            "GRI", "quantitative", "ratio"
        )
        
        self.add_indicator(
            "WomenInLeadership", "Social", "Diversity & Inclusion",
            "Percentage of women in leadership positions",
            "SASB", "quantitative", "%"
        )
        
        # Health & Safety (GRI 403)
        self.add_indicator(
            "WorkplaceInjuries", "Social", "Health & Safety",
            "Work-related injuries and injury rates",
            "GRI", "quantitative", "rate per hours worked"
        )
        
        self.add_indicator(
            "OccupationalDiseases", "Social", "Health & Safety",
            "Work-related ill health and occupational diseases",
            "GRI", "quantitative", "cases"
        )
        
        self.add_indicator(
            "SafetyTraining", "Social", "Health & Safety",
            "Workers covered by occupational health and safety management system",
            "GRI", "quantitative", "%"
        )
        
        # Training & Development (GRI 404)
        self.add_indicator(
            "EmployeeTraining", "Social", "Training & Development",
            "Average hours of training per year per employee",
            "GRI", "quantitative", "hours"
        )
        
        self.add_indicator(
            "SkillsDevelopment", "Social", "Training & Development",
            "Programs for upgrading employee skills and transition assistance",
            "GRI", "qualitative", "description"
        )
        
        # Human Rights (GRI 412)
        self.add_indicator(
            "HumanRightsAssessment", "Social", "Human Rights",
            "Operations subject to human rights reviews or impact assessments",
            "GRI", "quantitative", "number and %"
        )
        
        self.add_indicator(
            "ChildLaborRisk", "Social", "Human Rights",
            "Operations and suppliers at significant risk for child labor",
            "GRI", "qualitative", "description"
        )
        
        # Community Relations (GRI 413)
        self.add_indicator(
            "CommunityEngagement", "Social", "Community Relations",
            "Operations with local community engagement and development programs",
            "GRI", "quantitative", "%"
        )
        
        self.add_indicator(
            "LocalSuppliers", "Social", "Community Relations",
            "Proportion of spending on local suppliers",
            "GRI", "quantitative", "%"
        )
        
        # Customer Privacy & Data Security (SASB)
        self.add_indicator(
            "DataSecurity", "Social", "Data Privacy",
            "Description of approach to identifying and addressing data security risks",
            "SASB", "qualitative", "description"
        )
        
        self.add_indicator(
            "CustomerPrivacy", "Social", "Data Privacy",
            "Number of data breaches and customers affected",
            "SASB", "quantitative", "number"
        )
        
    def add_governance_indicators(self):
        """Add comprehensive governance indicators from GRI and SASB"""
        
        # Corporate Governance (GRI 102)
        self.add_indicator(
            "BoardIndependence", "Governance", "Board Composition",
            "Percentage of independent board members",
            "GRI", "quantitative", "%"
        )
        
        self.add_indicator(
            "BoardMeetings", "Governance", "Board Composition",
            "Number of board meetings per year and attendance rates",
            "GRI", "quantitative", "number"
        )
        
        self.add_indicator(
            "ExecutiveCompensation", "Governance", "Executive Compensation",
            "Annual total compensation ratio and percentage increase",
            "GRI", "quantitative", "ratio"
        )
        
        # Ethics & Compliance (GRI 205, 206)
        self.add_indicator(
            "AntiCorruption", "Governance", "Ethics & Compliance",
            "Operations assessed for risks related to corruption",
            "GRI", "quantitative", "number and %"
        )
        
        self.add_indicator(
            "EthicsTraining", "Governance", "Ethics & Compliance",
            "Communication and training about anti-corruption policies",
            "GRI", "quantitative", "%"
        )
        
        self.add_indicator(
            "LegalCompliance", "Governance", "Ethics & Compliance",
            "Legal actions for anti-competitive behavior and violations",
            "GRI", "quantitative", "number"
        )
        
        # Risk Management (TCFD)
        self.add_indicator(
            "ClimateRiskGovernance", "Governance", "Risk Management",
            "Board oversight of climate-related risks and opportunities",
            "TCFD", "qualitative", "description"
        )
        
        self.add_indicator(
            "RiskManagementProcess", "Governance", "Risk Management",
            "Processes for identifying, assessing and managing climate-related risks",
            "TCFD", "qualitative", "description"
        )
        
        # Supply Chain Management
        self.add_indicator(
            "SupplierAssessment", "Governance", "Supply Chain",
            "New suppliers screened using environmental and social criteria",
            "GRI", "quantitative", "%"
        )
        
        self.add_indicator(
            "SupplierAudits", "Governance", "Supply Chain",
            "Suppliers assessed for environmental and social impacts",
            "GRI", "quantitative", "number"
        )
        
        # Transparency & Reporting
        self.add_indicator(
            "ESGReporting", "Governance", "Transparency",
            "Publication of sustainability reports and ESG disclosures",
            "GRI", "qualitative", "yes/no"
        )
        
        self.add_indicator(
            "StakeholderEngagement", "Governance", "Transparency",
            "Stakeholder groups engaged and key topics raised",
            "GRI", "qualitative", "description"
        )
        
    def generate_comprehensive_set(self):
        """Generate the complete expanded ESG indicator set"""
        print("Generating comprehensive ESG indicator set...")
        
        # Load original indicators
        self.load_original_indicators()
        
        # Add indicators from major frameworks
        self.add_environmental_indicators()
        self.add_social_indicators() 
        self.add_governance_indicators()
        
        print(f"Generated {len(self.indicators)} ESG indicators")
        
        # Summary by category
        categories = {}
        for indicator in self.indicators:
            cat = indicator['category']
            categories[cat] = categories.get(cat, 0) + 1
            
        print("\nIndicators by category:")
        for cat, count in categories.items():
            print(f"  {cat}: {count} indicators")
            
        return self.indicators
        
    def save_to_json(self, filename: str = "expanded_esg_indicators.json"):
        """Save expanded indicators to JSON file"""
        output = {
            "metadata": {
                "total_indicators": len(self.indicators),
                "generation_date": datetime.now().isoformat(),
                "sources": self.sources,
                "categories": list(set(ind['category'] for ind in self.indicators))
            },
            "indicators": self.indicators
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved expanded indicators to {filename}")
        
    def save_to_csv(self, filename: str = "expanded_esg_indicators.csv"):
        """Save expanded indicators to CSV file"""
        if not self.indicators:
            return
            
        fieldnames = self.indicators[0].keys()
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.indicators)
        print(f"Saved expanded indicators to {filename}")
        
    def create_annotation_guidelines(self, filename: str = "esg_annotation_guidelines.md"):
        """Create annotation guidelines for the expanded indicator set"""
        guidelines = f"""# ESG Indicator Annotation Guidelines

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Indicators: {len(self.indicators)}

## Overview

This document provides annotation guidelines for extracting {len(self.indicators)} ESG indicators from corporate reports using the expanded indicator set based on major ESG frameworks.

## Sources

"""
        
        for source, reference in self.sources.items():
            guidelines += f"- **{source}**: {reference}\n"
            
        guidelines += "\n## Indicator Categories\n\n"
        
        # Group by category
        by_category = {}
        for indicator in self.indicators:
            cat = indicator['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(indicator)
            
        for category, indicators in by_category.items():
            guidelines += f"### {category} ({len(indicators)} indicators)\n\n"
            
            # Group by subcategory
            by_subcat = {}
            for ind in indicators:
                subcat = ind['subcategory']
                if subcat not in by_subcat:
                    by_subcat[subcat] = []
                by_subcat[subcat].append(ind)
                
            for subcat, subcat_indicators in by_subcat.items():
                guidelines += f"#### {subcat}\n\n"
                
                for ind in subcat_indicators:
                    guidelines += f"**{ind['name']}**\n"
                    guidelines += f"- Description: {ind['description']}\n"
                    guidelines += f"- Source: {ind['source']}\n"
                    guidelines += f"- Type: {ind['metric_type']}\n"
                    if ind['unit']:
                        guidelines += f"- Unit: {ind['unit']}\n"
                    guidelines += "\n"
                    
        guidelines += """\n## Annotation Instructions\n\n1. **Text Identification**: Look for sentences or paragraphs that mention or discuss any of the above indicators
2. **Context Extraction**: Extract the surrounding context that provides meaning to the indicator
3. **Value Extraction**: When possible, extract quantitative values, targets, or qualitative assessments
4. **Source Attribution**: Note the section/page where the indicator was found
5. **Confidence Scoring**: Rate confidence in the extraction (High/Medium/Low)

## Example Annotations

```json
{
  "indicator": "GHGEmissionsScope1",
  "text": "Our direct emissions decreased by 15% to 2.3 million tonnes CO2e in 2023",
  "value": "2.3 million tonnes CO2e",
  "trend": "decreased by 15%",
  "year": "2023",
  "confidence": "High",
  "source_section": "Environmental Performance"
}
```
"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(guidelines)
        print(f"Created annotation guidelines: {filename}")

def main():
    """Main execution function"""
    expander = ESGIndicatorExpansion()
    
    # Generate comprehensive indicator set
    indicators = expander.generate_comprehensive_set()
    
    # Save outputs
    expander.save_to_json("data/expanded_esg_indicators.json")
    expander.save_to_csv("data/expanded_esg_indicators.csv")
    expander.create_annotation_guidelines("docs/esg_annotation_guidelines.md")
    
    print("\n=== ESG Indicator Expansion Complete ===")
    print(f"Expanded from 2 to {len(indicators)} indicators")
    print("Files created:")
    print("  - data/expanded_esg_indicators.json")
    print("  - data/expanded_esg_indicators.csv")
    print("  - docs/esg_annotation_guidelines.md")
    
if __name__ == "__main__":
    main()