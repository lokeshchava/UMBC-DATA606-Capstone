 # 1. Vehicle insurance fraud detection and analysis
- Vehicle insurance fraud detection and analysis
- Prepared for UMBC Data Science Master Degree Capstone by Lokesh Chava under the guidance of Dr Chaojie (Jay) Wang
- Author Name: Lokesh Chava
- GitHub profile: https://github.com/lokeshchava
- LinkedIn profile: https://www.linkedin.com/in/chavalokesh
- PowerPoint presentation file: https://docs.google.com/presentation/d/1K-_GsqiDmS10ktd7dYoX4rYIWv3ppP3L7I3vTkyYeag/edit?usp=sharing
    
# 2. Background
Insurance Fraud bumps up to millions of dollars every year for insurance companies. it is very essential to have a fraud detection system in place to avoid fraudulent claims and maintain fairness in the claim process, thus enhancing the company's reputation and building trust.
- **What is it about?**  
  A fraud detection system for Vehicle insurance claims helps in identifying fraudulent requests for insurance claims and helps in improving a very robust and autonomous system.  
- **Why does it matter?**  
  Fraudulent claims introduce additional risk to insurance portfolios. By detecting and preventing fraud, insurance companies can better manage their risk exposure and maintain the financial health of the organization.  
- **What are your research questions?**  
  How to classify the fair and fraud requests?   
  what are the factors that determine fraud activities?  
  which algorithm best suits the model?  
# 3. Data 

Describe the datasets you are using to answer your research questions.

- Data sources: [Kaggle Link](https://www.kaggle.com/datasets/khusheekapoor/vehicle-insurance-fraud-detection)
- Data size: 3.69 MB
- Data shape: 15420 rows, 33 columns
- Data dictionary
   - Integer type columns: WeekOfMonth, WeekOfMonthClaimed, Age, PolicyNumber, RepNumber, Deductible, DriverRating, Year
   - Object type Columns: Month, DayOfWeek, Make, AccidentArea, DayOfWeekClaimed, MonthClaimed, Sex, MaritalStatus, Fault, PolicyType, VehicleCategory, VehiclePrice,
       Days:Policy-Accident, Days:Policy-Claim, PastNumberOfClaims, AgeOfVehicle, AgeOfPolicyHolder, PoliceReportFiled, WitnessPresent, AgentType,
       NumberOfSuppliments, AddressChange-Claim, NumberOfCars, BasePolicy, FraudFound
- variable used as target/label in ML model
  - Claim Level Fraud Detection:
    - Target Variable: FraudFound. These will indicate whether a claim is fraudulent or not.  
  
- Columns selected as features/predictors for your ML models
  - Claim Level Fraud Detection:
    - Features can include various attributes related to the claim, customer, and incident:
      - Customer demographics (AGE, MARITAL_STATUS, SOCIAL_CLASS, etc.).
      - Claim details (CLAIM_AMOUNT, INCIDENT_SEVERITY, INCIDENT_STATE, etc.).
      - Incident details (DayOfWeek, PoliceReportFiled, NumberOfCars, etc.).

# 4. Project Flow
  - Initial data preparation
  - Performing Exploratory Data Analysis
  - Selecting Data Model
  - Hypertuning the model parameters
  - Testing
  - Web Deployment using Streamlit

# 5. Exploratory Data Analysis (EDA)
  - Initial Statistics:  
|    |WeekOfMonth	|WeekOfMonthClaimed	|Age	|PolicyNumber	|RepNumber	|Deductible	|DriverRating	|Year |
------------------------------------------------------------------------------------------------------
|count	|15420.000000	|15420.000000	|15420.000000	|15420.000000	|15420.000000	|15420.000000	|15420.000000	|15420.000000 |
mean	2.788586	2.693969	39.855707	7710.500000	8.483268	407.704280	2.487808	1994.866472
std	1.287585	1.259115	13.492377	4451.514911	4.599948	43.950998	1.119453	0.803313
min	1.000000	1.000000	0.000000	1.000000	1.000000	300.000000	1.000000	1994.000000
25%	2.000000	2.000000	31.000000	3855.750000	5.000000	400.000000	1.000000	1994.000000
50%	3.000000	3.000000	38.000000	7710.500000	8.000000	400.000000	2.000000	1995.000000
75%	4.000000	4.000000	48.000000	11565.250000	12.000000	400.000000	3.000000	1996.000000
max	5.000000	5.000000	80.000000	15420.000000	16.000000	700.000000	4.000000	1996.000000
