import os

import streamlit as st
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType
import findspark
findspark.init()

# Creating Session
spark = SparkSession.builder.appName("ClassificationwithSpark").getOrCreate()

model = RandomForestClassificationModel.load("random_forest")


def loan_prediction(input_data):
    data = [
        (
            float(input_data[0]),
            float(input_data[1]),
            float(input_data[2]),
            float(input_data[3]),
            int(input_data[4]),
            float(input_data[5]),
            float(input_data[6]),
            float(input_data[7]),
            float(input_data[8]),
            float(input_data[9]),
            
        ),
    ]
    schema = StructType(
        [   
            StructField("Gender_", FloatType(), True),
            StructField("Dependents_", FloatType(), True),
            StructField("Education_", FloatType(), True),
            StructField("Self_Employed_", FloatType(), True),
            StructField("ApplicantIncome", IntegerType(), True),
            StructField("LoanAmount", FloatType(), True),
            StructField("Credit_History", FloatType(), True),
            StructField("Property_Area_", FloatType(), True),
            StructField("Total_Income", FloatType(), True),
            StructField("Loan_Status_", FloatType(), True),    
        ]
    )

    single_row_df = spark.createDataFrame(data, schema)

    # Assuming "Outcome" is the label column and "features" is the feature vector
    vector_assembler = VectorAssembler(
        inputCols=[
            "Gender_",
            "Dependents_",
            "Education_",
            "Self_Employed_",
            "ApplicantIncome",
            "LoanAmount",
            "Credit_History",
            "Property_Area_",
            "Total_Income",
            "Loan_Status_"
        ],
        outputCol="features",
    )

    single_row_df = vector_assembler.transform(single_row_df)

    # Use the trained model to make predictions
    prediction = model.transform(single_row_df)

    # Display the prediction
    result = prediction.select("features", "rawPrediction", "probability", "prediction").rdd.flatMap(lambda x: x).collect()
    if result[-1] == 0.0:
        return "Loan Not Approved"
    else:
        return "Loan Approved"

def main():
    # giving a title
    st.title("Loan Approve Prediction Web App")

    # getting the input data from the user

    Gender = st.text_input("Gender")
    Dependents = st.text_input("Dependents")
    Education = st.text_input("Education")
    Self_Employed = st.text_input("Self_Employed")
    ApplicantIncome = st.text_input("Applicant Income")
    LoanAmount = st.text_input("Loan Amount")
    Credit_History = st.text_input("Credit_History")
    Property_Area = st.text_input("Property_Area_")
    Total_Income = st.text_input("Total_Income")
    Loan_Status = st.text_input("Loan_Status_")
    
    

    # code for Prediction
    diagnosis = ""

    # creating a button for Prediction

    if st.button("Loan Prediction Result"):
        diagnosis = loan_prediction(
            [
                Gender,
                Dependents,
                Education,
                Self_Employed,
                LoanAmount,
                ApplicantIncome,
                Credit_History,
                Property_Area,
                Total_Income,
                Loan_Status
                
            ]
        )

    st.success(diagnosis)


if __name__ == "__main__":
    main()

