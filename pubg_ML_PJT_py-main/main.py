import data_io
import preprocess_dataset
import learn_model

df_train = data_io.load_dataset("train_V2.csv")

print(df_train)
print(df_train.columns)
df_train = preprocess_dataset.preprocess(df_train)
print(df_train)
print(df_train.columns)
model = learn_model.linear_reg(df_train)
model_2 = learn_model.poly_reg(df_train)

