import streamlit as st
from pathlib import Path
import base64
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Initial page config

st.set_page_config(
     page_title='Data and Decision Making Demo',
     layout="wide",
     initial_sidebar_state="expanded",
)

def get_droplist(dataset, columns, items):
    droplst = []
    for i in range(len(items)):
        if items[i] != True:
            for j in range(len(dataset)):
                if dataset["question"][j] == columns[i]:
                    droplst.append(j)
    return droplst
    
def get_log_regression_dataset(dataset, columns, values, ylabel):
    kept_list = []
    for i in range(len(columns)):
        if values[i] == True:
            kept_list.append(columns[i])
    
    return dataset[kept_list], dataset[ylabel]


def main():
    df = pd.read_csv('ND and California Climate Opinion Data (Howe et al, 2006)-2.csv')
    viz = pd.read_csv('State Plotly Visualization.csv')
    metadata = pd.read_csv('howe-2016-metadata-updated.csv')
    st.title('Comparing North Dakotan and California Climate Change Attitudes')
    st.write("---") 
    st.subheader('Contained in this page is extensive survey data from 2015-16 comparing opinions relevant to climate change of different counties within California and North Dakota.')
    
    st.write('I wanted to see if opinions in these different categories were too out of touch as to where I could ask someone from some California counties these questions, and their responses would be similar to those from North Dakota. What fields are too biased given geograghical and situational context, and what fields should I care about?')
    st.write('Another food for thought: just because you can separate data, does it mean you should? And in what meaning?')
   
    st.write("---")
    st.write("For these visualizations, we will be working with training data of a large sample of counties from each state. Here is example data to view: ")
    st.subheader("California Data Header")
    st.table(data=df.head())
    st.subheader("North Dakota Data Tail")
    st.table(data=df.tail())
    st.subheader("Relevant Metadata")
    st.table(data=metadata)
    st.write("---")
    st.subheader("Model Building")
    columnlst = ['discuss', 'CO2limits', 'trustclimsciSST', 'regulate', 'supportRPS', 'fundrenewables', 'happening', 'human', 'consensus', 'worried', 'personal', 'harmUS', 'devharm', 'futuregen', 'harmplants', 'timing']
        
    with st.form("my_form"):
        st.write("Choose what variables you'd like in your model: ")
        form_items = [st.checkbox(col) for col in columnlst]
        submit = st.form_submit_button()

    if submit:
        droplst = get_droplist(viz, columnlst, form_items)
        filtered_viz = viz.drop(droplst).reset_index(drop=True)

        xdata, ydata = get_log_regression_dataset(df, columnlst, form_items, "StateDiff")
        clf = LogisticRegression(random_state=42).fit(xdata, ydata)
        base_pred = clf.predict(xdata)
        base_pred_prob = clf.predict_proba(xdata)[:, 1]


        st.subheader("Box Plots Comparing By Field")

        fig = px.box(filtered_viz, x="question", y="score", color="state", color_discrete_map={"blue": "lime", "green": "pink"})
        fig.update_traces(quartilemethod="exclusive")
        fig.update_layout(width=1300)
        st.plotly_chart(fig)


        predicted_df = pd.DataFrame(list(zip(df["StateDiff"], base_pred_prob, df["stateName"])), columns=["Real State", "Predicted State (0 = California, 1 = North Dakota)", "State Name"])

        fig2 = px.box(predicted_df, x="State Name", y="Predicted State (0 = California, 1 = North Dakota)", color="State Name", color_discrete_map={"blue": "lime", "green": "pink"})
        fig2.update_traces(quartilemethod="exclusive")
        fig2.update_layout(width=1300)
        st.plotly_chart(fig2)
        st.caption("More extensive confusion matrix on training set comparing Logistic Regression Results, where 0 is California and 1 is North Dakota. Are they separated? Is that a good thing? Would have a different model been better? What fields missing from the visualization are worth improving?")

        st.subheader("Confusion Matrix from a Logistic Regression Model")
        confusionmatrix = confusion_matrix(ydata,base_pred)
        st.write(confusionmatrix)
        st.caption("Confusion Matrix on training set where 0 is California and 1 is North Dakota.")



    
    st.write("---")
    st.write("Data Manipulated From: ")
    st.write("Howe, P. D., Mildenberger, M., Marlon, J. R., & Leiserowitz, A. (2015). Geographic variation in opinions on climate change at state and local scales in the USA. Nature Climate Change, 5(6). https://doi.org/10.1038/nclimate2583")

    return None

# Thanks to streamlitopedia for the following code snippet

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded



# Run main()

if __name__ == '__main__':
    main()