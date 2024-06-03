from train_model import *
# Sample DataFrame
# data = {'column_name': [1, 2, 2, 3, 4, 4, 5]}
# df = pd.DataFrame(data)
#
# # Check for duplicated values
#
# # if not check_unique(df['column_name']):
# #     print(get_unique(df['column_name']))
# citizenship_status = [
#     "ByBirth", "ByBirth", "ByBirth", "ByBirth", "ByOtherMeans", "ByBirth", "ByBirth", "ByBirth",
#     "ByBirth", "ByBirth", "ByBirth", "ByBirth", "ByBirth", "ByBirth", "ByBirth", "ByBirth",
#     "ByBirth", "ByBirth", "ByBirth", "ByBirth", "ByBirth", "ByBirth", "ByOtherMeans", "ByBirth",
#     "ByBirth", "ByBirth", "ByBirth", "ByBirth", "ByBirth", "ByBirth", "ByBirth"
# ]
# df1 = pd.DataFrame(citizenship_status, columns=["Citizen"])
#
# #print(encode_columns(df1["Citizen"]))
#
#
# ethnicities = [
#     "White", "Black", "Black", "White", "White", "White", "Black", "White",
#     "Black", "White", "Black", "Black", "White", "White", "White", "White",
#     "White", "White", "Black", "Black", "White", "White", "White", "Black",
#     "White", "Black", "Asian", "Asian", "Black", "White", "White", "Black",
#     "Asian", "White", "White"
# ]
#
# # Convert list to DataFrame
# df2 = pd.DataFrame({"Ethnicity": ethnicities})
#
# print(encode_columns(df2["Ethnicity"]))

# train_model("clean_dataset.csv", "Approved", "clean_dataset.csv", "credit_card_model")
# train_model("trainset.csv", "Subscribed", "testset.csv", "term_deposit_model")
#
# train_model("Airline_customer_satisfaction.csv", "satisfaction", "Airline_customer_satisfaction.csv", "airline_model")

# print(get_mapping_real_value(load_data("trainset.csv"), load("term_deposit_model/term_deposit_model_encoded_df.joblib"), "Subscribed", "1"))

single_data_point = {
    'age': 41,
    'job': 'blue-collar',
    'marital': 'divorced',
    'education': 'basic.4y',
    'housing': 'yes',
    'loan': 'no',
    'contact': 'telephone',
    'month': 'may',
    'day_of_week': 'mon',
    'duration': 1575,
    'campaign': 1,
    'pdays': 999,
    'poutcome': 'nonexistent',
    'nr.employed': 5191,
}

single_data_point2 = {
    'age': 43,
    'job': 'management',
    'marital': 'married',
    'education': 'professional.course',
    'housing': 'yes',
    'loan': 'no',
    'contact': 'telephone',
    'month': 'may',
    'day_of_week': 'tue',
    'duration': 310,
    'campaign': 1,
    'pdays': 999,
    'poutcome': 'nonexistent',
    'nr.employed': 5191,
    'Subscribed': 'no'
}

single_data_point_test_1 = {
    'age': 49,
    'job': 'admin.',
    'marital': 'single',
    'education': 'high.school',
    'housing': 'yes',
    'loan': 'no',
    'contact': 'cellular',
    'month': 'oct',
    'day_of_week': 'fri',
    'duration': 136,
    'campaign': 2,
    'pdays': 999,
    'poutcome': 'nonexistent',
    'nr.employed': 5017.5,
}

single_data_point_test_1 = {
    'age': 36,
    'job': 'admin.',
    'marital': 'married',
    'education': 'university.degree',
    'housing': 'no',
    'loan': 'no',
    'contact': 'cellular',
    'month': 'oct',
    'day_of_week': 'fri',
    'duration': 342,
    'campaign': 1,
    'pdays': 999,
    'poutcome': 'failure',
    'nr.employed': 5017.5,
}

# df = pd.read_csv("testset.csv")
#
# # Define the test_model function
#
# # Iterate over the rows of the DataFrame
# for index, row in df.iterrows():
#     if index < 10:
#         # Extract the single data point for the current row
#         print(row)
#         test_model("term_deposit_model", single_data_point, "Subscribed")
#         print("----------------------------------------------------------------------------------")

single_data_point_ss = {
    'age': '62',
    'job': 'retired',
    'marital': 'married',
    'education': 'university.degree',
    'housing': 'no',
    'loan': 'no',
    'contact': 'cellular',
    'month': 'oct',
    'day_of_week': 'fri',
    'duration': '717',
    'campaign': '2',
    'pdays': '999',
    'poutcome': 'nonexistent',
    'nr.employed': '5017.5',
}
row_values = {
    'Customer Type': 'Loyal Customer',
    'Age': '65',
    'Type of Travel': 'Personal Travel',
    'Class': 'Eco',
    'Flight Distance': '265',
    'Seat comfort': '0',
    'Departure/Arrival time convenient': '0',
    'Food and drink': '0',
    'Gate location': '2',
    'Inflight wifi service': '2',
    'Inflight entertainment': '4',
    'Online support': '2',
    'Ease of Online booking': '3',
    'On-board service': '3',
    'Leg room service': '0',
    'Baggage handling': '3',
    'Checkin service': '5',
    'Cleanliness': '3',
    'Online boarding': '2',
    'Departure Delay in Minutes': '0',
    'Arrival Delay in Minutes': '0'
}

data_point = {
    "age": 41,
    "job": "blue-collar",
    "marital": "divorced",
    "education": "basic.4y",
    "housing": "yes",
    "loan": "no",
    "contact": "telephone",
    "month": "may",
    "day_of_week": "mon",
    "duration": 1575,
    "campaign": 1,
    "pdays": 999,
    "poutcome": "nonexistent",
    "nr.employed": 5191
}
# test_model("airline_model", row_values, "satisfaction")


data = {
    'age': [41, 49, 49, 41, 45, 42],
    'job': ['blue-collar', 'entrepreneur', 'technician', 'technician', 'blue-collar', 'blue-collar'],
    'marital': ['divorced', 'married', 'married', 'married', 'married', 'married'],
    'education': ['basic.4y', 'university.degree', 'basic.9y', 'professional.course', 'basic.9y', 'basic.9y'],
    'housing': ['yes', 'yes', 'no', 'yes', 'yes', 'yes'],
    'loan': ['no', 'no', 'no', 'no', 'no', 'yes'],
    'contact': ['telephone', 'telephone', 'telephone', 'telephone', 'telephone', 'telephone'],
    'month': ['may', 'may', 'may', 'may', 'may', 'may'],
    'day_of_week': ['mon', 'mon', 'mon', 'mon', 'mon', 'mon'],
    'duration': [1575, 1042, 1467, 579, 461, 673],
    'campaign': [1, 1, 1, 1, 1, 2],
    'pdays': [999, 999, 999, 999, 999, 999],
    'poutcome': ['nonexistent', 'nonexistent', 'nonexistent', 'nonexistent', 'nonexistent', 'nonexistent'],
    'nr.employed': [5191, 5191, 5191, 5191, 5191, 5191],
    'Subscribed': ['yes', 'yes', 'yes', 'yes', 'yes', 'yes']
}

print("Accuracy ############# "+str(train_model("bank-full.csv", "y", "bank-num.csv", "bank_saransh", 1)))


single_data_point = {
    'Sorry_end': 2,
    'Ignore_diff': 2,
    'begin_correct': 4,
    'Contact': 1,
    'Special_time': 0,
    'No_home_time': 0,
    '2_strangers': 0,
    'enjoy_holiday': 0,
    'enjoy_travel': 0,
    'common_goals': 0,
    'harmony': 1,
    'freeom_value': 0,
    'entertain': 1,
    'people_goals': 1,
    'dreams': 0,
    'love': 1,
    'happy': 0,
    'marriage': 0,
    'roles': 0,
    'trust': 1,
    'likes': 0,
    'care_sick': 0,
    'fav_food': 0,
    'stresses': 0,
    'inner_world': 0,
    'anxieties': 0,
    'current_stress': 0,
    'hopes_wishes': 0,
    'know_well': 0,
    'friends_social': 1,
    'Aggro_argue': 1,
    'Always_never': 2,
    'negative_personality': 1,
    'offensive_expressions': 2,
    'insult': 0,
    'humiliate': 1,
    'not_calm': 2,
    'hate_subjects': 1,
    'sudden_discussion': 3,
    'idk_what\'s_going_on': 3,
    'calm_breaks': 2,
    'argue_then_leave': 1,
    'silent_for_calm': 1,
    'good_to_leave_home': 2,
    'silence_instead_of_discussion': 3,
    'silence_for_harm': 2,
    'silence_fear_anger': 1,
    'I\'m_right': 3,
    'accusations': 3,
    'I\'m_not_guilty': 3,
    'I\'m_not_wrong': 2,
    'no_hesitancy_inadequate': 3,
    'you\'re_inadequate': 2,
    'incompetence': 1
}


# test_model("divorce_model", single_data_point, "Divorce_Y_N")