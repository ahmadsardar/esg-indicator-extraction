# ESG Indicator Annotation Guidelines

Generated: 2025-08-18 17:43:28
Total Indicators: 46

## Overview

This document provides annotation guidelines for extracting 46 ESG indicators from corporate reports using the expanded indicator set based on major ESG frameworks.

## Sources

- **GRI**: Global Reporting Initiative Standards - https://www.globalreporting.org/standards
- **SASB**: Sustainability Accounting Standards Board - https://sasb.ifrs.org/standards/
- **TCFD**: Task Force on Climate-related Financial Disclosures
- **EU_TAXONOMY**: EU Taxonomy for Sustainable Activities
- **SDG**: UN Sustainable Development Goals
- **ORIGINAL**: Original esgontology.owl file

## Indicator Categories

### Environmental (18 indicators)

#### Energy Management

**EnergyEfficiency**
- Description: Measures related to energy efficiency improvements and energy consumption optimization
- Source: ORIGINAL
- Type: quantitative
- Unit: kWh/unit or %

**EnergyConsumption**
- Description: Total energy consumption within the organization
- Source: GRI
- Type: quantitative
- Unit: GJ or MWh

**RenewableEnergyPercentage**
- Description: Percentage of energy consumption from renewable sources
- Source: GRI
- Type: quantitative
- Unit: %

**EnergyIntensity**
- Description: Energy consumption per unit of activity or revenue
- Source: GRI
- Type: quantitative
- Unit: GJ/revenue

#### Waste Management

**WasteReductionIndicator**
- Description: Indicators measuring waste reduction efforts and circular economy practices
- Source: ORIGINAL
- Type: quantitative
- Unit: tons or %

**WasteGenerated**
- Description: Total weight of waste generated
- Source: GRI
- Type: quantitative
- Unit: tons

**WasteRecycled**
- Description: Percentage of waste diverted from disposal through recycling
- Source: GRI
- Type: quantitative
- Unit: %

**HazardousWaste**
- Description: Total weight of hazardous waste generated
- Source: GRI
- Type: quantitative
- Unit: tons

#### Climate Change

**GHGEmissionsScope1**
- Description: Direct greenhouse gas emissions from owned or controlled sources
- Source: GRI
- Type: quantitative
- Unit: tCO2e

**GHGEmissionsScope2**
- Description: Indirect greenhouse gas emissions from purchased energy
- Source: GRI
- Type: quantitative
- Unit: tCO2e

**GHGEmissionsScope3**
- Description: Other indirect greenhouse gas emissions in value chain
- Source: GRI
- Type: quantitative
- Unit: tCO2e

**CarbonIntensity**
- Description: Carbon emissions per unit of revenue or production
- Source: SASB
- Type: quantitative
- Unit: tCO2e/revenue

#### Water Management

**WaterConsumption**
- Description: Total water consumption by the organization
- Source: GRI
- Type: quantitative
- Unit: m³

**WaterRecycling**
- Description: Percentage of water recycled and reused
- Source: GRI
- Type: quantitative
- Unit: %

**WaterStressAreas**
- Description: Water consumption in water-stressed areas
- Source: SASB
- Type: quantitative
- Unit: m³

#### Biodiversity

**BiodiversityImpact**
- Description: Significant impacts on biodiversity in protected areas
- Source: GRI
- Type: qualitative
- Unit: description

**LandUse**
- Description: Total land area owned, leased, or managed for production activities
- Source: SASB
- Type: quantitative
- Unit: hectares

#### Air Quality

**AirEmissions**
- Description: Air emissions of pollutants (NOx, SOx, particulates)
- Source: SASB
- Type: quantitative
- Unit: tons

### Social (16 indicators)

#### Employment

**EmployeeTurnover**
- Description: Total number and rates of new employee hires and employee turnover
- Source: GRI
- Type: quantitative
- Unit: rate %

**EmployeeBenefits**
- Description: Benefits provided to full-time employees
- Source: GRI
- Type: qualitative
- Unit: description

#### Diversity & Inclusion

**BoardDiversity**
- Description: Diversity of governance bodies and employees by gender, age, minority groups
- Source: GRI
- Type: quantitative
- Unit: %

**GenderPayGap**
- Description: Ratio of basic salary and remuneration of women to men
- Source: GRI
- Type: quantitative
- Unit: ratio

**WomenInLeadership**
- Description: Percentage of women in leadership positions
- Source: SASB
- Type: quantitative
- Unit: %

#### Health & Safety

**WorkplaceInjuries**
- Description: Work-related injuries and injury rates
- Source: GRI
- Type: quantitative
- Unit: rate per hours worked

**OccupationalDiseases**
- Description: Work-related ill health and occupational diseases
- Source: GRI
- Type: quantitative
- Unit: cases

**SafetyTraining**
- Description: Workers covered by occupational health and safety management system
- Source: GRI
- Type: quantitative
- Unit: %

#### Training & Development

**EmployeeTraining**
- Description: Average hours of training per year per employee
- Source: GRI
- Type: quantitative
- Unit: hours

**SkillsDevelopment**
- Description: Programs for upgrading employee skills and transition assistance
- Source: GRI
- Type: qualitative
- Unit: description

#### Human Rights

**HumanRightsAssessment**
- Description: Operations subject to human rights reviews or impact assessments
- Source: GRI
- Type: quantitative
- Unit: number and %

**ChildLaborRisk**
- Description: Operations and suppliers at significant risk for child labor
- Source: GRI
- Type: qualitative
- Unit: description

#### Community Relations

**CommunityEngagement**
- Description: Operations with local community engagement and development programs
- Source: GRI
- Type: quantitative
- Unit: %

**LocalSuppliers**
- Description: Proportion of spending on local suppliers
- Source: GRI
- Type: quantitative
- Unit: %

#### Data Privacy

**DataSecurity**
- Description: Description of approach to identifying and addressing data security risks
- Source: SASB
- Type: qualitative
- Unit: description

**CustomerPrivacy**
- Description: Number of data breaches and customers affected
- Source: SASB
- Type: quantitative
- Unit: number

### Governance (12 indicators)

#### Board Composition

**BoardIndependence**
- Description: Percentage of independent board members
- Source: GRI
- Type: quantitative
- Unit: %

**BoardMeetings**
- Description: Number of board meetings per year and attendance rates
- Source: GRI
- Type: quantitative
- Unit: number

#### Executive Compensation

**ExecutiveCompensation**
- Description: Annual total compensation ratio and percentage increase
- Source: GRI
- Type: quantitative
- Unit: ratio

#### Ethics & Compliance

**AntiCorruption**
- Description: Operations assessed for risks related to corruption
- Source: GRI
- Type: quantitative
- Unit: number and %

**EthicsTraining**
- Description: Communication and training about anti-corruption policies
- Source: GRI
- Type: quantitative
- Unit: %

**LegalCompliance**
- Description: Legal actions for anti-competitive behavior and violations
- Source: GRI
- Type: quantitative
- Unit: number

#### Risk Management

**ClimateRiskGovernance**
- Description: Board oversight of climate-related risks and opportunities
- Source: TCFD
- Type: qualitative
- Unit: description

**RiskManagementProcess**
- Description: Processes for identifying, assessing and managing climate-related risks
- Source: TCFD
- Type: qualitative
- Unit: description

#### Supply Chain

**SupplierAssessment**
- Description: New suppliers screened using environmental and social criteria
- Source: GRI
- Type: quantitative
- Unit: %

**SupplierAudits**
- Description: Suppliers assessed for environmental and social impacts
- Source: GRI
- Type: quantitative
- Unit: number

#### Transparency

**ESGReporting**
- Description: Publication of sustainability reports and ESG disclosures
- Source: GRI
- Type: qualitative
- Unit: yes/no

**StakeholderEngagement**
- Description: Stakeholder groups engaged and key topics raised
- Source: GRI
- Type: qualitative
- Unit: description


## Annotation Instructions

1. **Text Identification**: Look for sentences or paragraphs that mention or discuss any of the above indicators
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
