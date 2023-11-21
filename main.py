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


def diabetes_prediction(input_data):
    data = [
        (
            int(input_data[0]),
            float(input_data[1]),
            float(input_data[2]),
            float(input_data[3]),
            float(input_data[4]),
            float(input_data[5]),
            float(input_data[6])
        ),
    ]
    schema = StructType(
        [
            StructField("ApplicantIncome", IntegerType(), True),
            StructField("LoanAmount", FloatType(), True),
            StructField("Gender_", FloatType(), True),
            StructField("Dependents_", FloatType(), True),
            StructField("Married_", FloatType(), True),
            StructField("Education_", FloatType(), True),
            StructField("Self_Employed_", FloatType(), True)
        ]
    )

    single_row_df = spark.createDataFrame(data, schema)

    # Assuming "Outcome" is the label column and "features" is the feature vector
    vector_assembler = VectorAssembler(
        inputCols=[
            "ApplicantIncome",
            "LoanAmount",
            "Gender_",
            "Dependents_",
            "Married_",
            "Education_",
            "Self_Employed_"
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

    ApplicantIncome = st.text_input("Applicant Income")
    LoanAmount = st.text_input("Loan Amount")
    Gender = st.text_input("Gender")
    Dependents = st.text_input("Dependents")
    Married = st.text_input("Marital Status")
    Education = st.text_input("Education")
    Self_Employed = st.text_input("Self_Employed")

    # code for Prediction
    diagnosis = ""

    # creating a button for Prediction

    if st.button("Loan Prediction Result"):
        diagnosis = diabetes_prediction(
            [
                ApplicantIncome,
                LoanAmount,
                Gender,
                Dependents,
                Married,
                Education,
                Self_Employed
            ]
        )

    st.success(diagnosis)


if __name__ == "__main__":
    main()

