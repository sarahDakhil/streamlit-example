import streamlit as st
import pandas as pd
import numpy as np
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy
from streamlit_card import card

st.title('📑Book Recommendation')

final_data = pd.read_csv(r'final_Data_book.csv')

# Convert the DataFrame to a Surprise dataset object
reader = Reader(rating_scale=(1, 10))  # Define the rating scale
data = Dataset.load_from_df(final_data[['User-ID', 'ISBN', 'Book-Rating']], reader)

# Split the dataset into training and test sets
trainset, testset = train_test_split(data, test_size=0.25)

'''
User-based collaborative recommendation system for books. Enter the user ID below.
'''

model = SVD()
model.fit(trainset)

# Generate recommendations for a target user

# Select a target user
target_user_id = st.text_input('🔑User ID')
if st.button("Get Recommendation"):
    if target_user_id.strip() != '':
        target_user_id = int(target_user_id)
        top_n = 5  # Number of recommendations to generate

        target_user_items = final_data[final_data['User-ID'] == target_user_id]
        predicted_ratings = [(item_id, model.predict(target_user_id, item_id).est) for item_id in target_user_items['ISBN']]
        predicted_ratings.sort(key=lambda x: x[1], reverse=True)
        recommended_items = [item_id for item_id, _ in predicted_ratings[:top_n]]

        # Print the ISBN, Book-Title, and Image-URL-L of the recommended books
        if recommended_items:
            st.header("Here's our picks for you ❤️", divider='orange')
            for item_id in recommended_items:
                book_info = final_data[final_data['ISBN'] == item_id][['ISBN', 'Book-Title', 'Image-URL-L']].values[0]
                hasClicked = card(
                    title=book_info[1],
                    text=' ',
                    image=book_info[2],
                    url=book_info[2],
                    styles={
                        "card": {
                            "width": "500px",
                            "height": "500px",
                            "border-radius": "60px",
                            "box-shadow": "0 0 10px rgba(0,0,0,0.5)",
                        }
                    }
                )
                st.write()
        else:
            st.write("haven't read any books 🐢...")
            st.header("Check out our top 5 book list🎈", divider='orange')
            Top_books = final_data.loc[final_data['Book-Rating'] > 8].sort_values(by='Book-Rating', ascending=False).sample(5, replace=False)
            for index, book in Top_books.iterrows():
                hasClicked = card(
                    title=book['Book-Title'],
                    text=' ',
                    image=book['Image-URL-L'],
                    url=book['Image-URL-L'],
                    styles={
                        "card": {
                            "width": "500px",
                            "height": "500px",
                            "border-radius": "60px",
                            "box-shadow": "0 0 10px rgba(0,0,0,0.5)",
                        }
                    }
                )
                st.write()
            
    else:
        st.write("Please enter a valid User ID.")

