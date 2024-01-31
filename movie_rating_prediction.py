import numpy as np
from tensorflow import keras
import tkinter as tk
import tkinter.ttk as ttk


def predictRating():
    movieModel = keras.models.load_model("Project4MovieModel")
    inputMovie = {"movie_title": np.array([titleEntry.get()]), "content_rating": np.array([contentRatingVal.get()]),
                  "genres": np.array([genresEntry.get()]), "directors": np.array([directorsEntry.get()]),
                  "authors": np.array([authorsEntry.get()]), "actors": np.array([actorsEntry.get()]),
                  "original_release_date": np.array([originalReleaseDateEntry.get()]), "streaming_release_date":
                      np.array([streamingReleaseDateEntry.get()]), "runtime": np.array([float(runtimeEntry.get())]),
                  "production_company": np.array([productionCoEntry.get()]), "tomatometer_count": np.array([57.0]),
                  "audience_count": np.array([143940.0]), "tomatometer_top_critics_count": np.array([20])}
    prediction = movieModel.predict(inputMovie)
    window.destroy()
    predictionWindow = tk.Tk()
    predictionWindow.title("Your Movie's Predicted Rating")
    predictionWindow.geometry("500x150")
    predictionWindow.configure(bg="Black")
    predictionHeader = ttk.Label(text="Your movie's predicted rating is: ", font=("Calibri", 25), background="black",
                                 foreground="#f0ca0e")
    predictionLabel = ttk.Label(text=str(round(prediction[0][0])), font=("Cooper Black", 30), background="black",
                                foreground="red")
    predictionHeader.pack()
    predictionLabel.pack()
    predictionWindow.mainloop()


window = tk.Tk()
contentRatingVal = tk.StringVar(value="G")
window.title("Movie Rating Predictor")
window.geometry("800x500")
window.configure(bg="black")
frame1 = tk.Frame(master=window, bg="black")
frame2 = tk.Frame(master=window, bg="black")
header = ttk.Label(text="Welcome to the movie rating predictor!", font=("Cooper Black", 20),
                   foreground="#f0ca0e", background="black")
instructions = ttk.Label(text="Please enter in all of the information for the movie whose rating you want to predict. "
                              "If you don't have any information for a specific section, leave it blank. "
                              "Then, click the \"Predict Rating\" button to get your prediction! If you need to enter"
                              " multiple items, such as with actors, please input your info as a comma-separated list. "
                              "Please enter dates in \"yyyy-mm-dd\" format.",
                         wraplength=400, font=("Calibri", 10), justify="center", foreground="white", background="black")
titleLabel = ttk.Label(master=frame1, text="Movie Title:", foreground="red", background="black")
titleEntry = ttk.Entry(master=frame1)
contentRatingLabel = ttk.Label(master=frame1, text="Content Rating:", foreground="red", background="black")
gRating = tk.Radiobutton(master=frame1, text="G", variable=contentRatingVal, value="G", foreground="red",
                         background="black")
pgRating = tk.Radiobutton(master=frame1, text="PG", variable=contentRatingVal, value="PG", foreground="red",
                          background="black")
pg13Rating = tk.Radiobutton(master=frame1, text="PG-13", variable=contentRatingVal, value="PG-13", foreground="red",
                            background="black")
rRating = tk.Radiobutton(master=frame1, text="R", variable=contentRatingVal, value="R", foreground="red",
                         background="black")
nrRating = tk.Radiobutton(master=frame1, text="NR", variable=contentRatingVal, value="NR", foreground="red",
                          background="black")
originalReleaseDateLabel = ttk.Label(master=frame1, text="Original Release Date:", foreground="red", background="black")
originalReleaseDateEntry = ttk.Entry(master=frame1)
streamingReleaseDateLabel = ttk.Label(master=frame1, text="Streaming Release Date:", foreground="red",
                                      background="black")
streamingReleaseDateEntry = ttk.Entry(master=frame1)
runtimeLabel = ttk.Label(master=frame1, text="Runtime (in minutes):", foreground="red", background="black")
runtimeEntry = ttk.Entry(master=frame1)
productionCoLabel = ttk.Label(master=frame2, text="Production Company:", foreground="red", background="black")
productionCoEntry = ttk.Entry(master=frame2)
genresLabel = ttk.Label(master=frame2, text="Genres:", foreground="red", background="black")
genresEntry = ttk.Entry(master=frame2, width=50)
directorsLabel = ttk.Label(master=frame2, text="Directors:", foreground="red", background="black")
directorsEntry = ttk.Entry(master=frame2, width=50)
authorsLabel = ttk.Label(master=frame2, text="Authors:", foreground="red", background="black")
authorsEntry = ttk.Entry(master=frame2, width=50)
actorsLabel = ttk.Label(master=frame2, text="Actors:", foreground="red", background="black")
actorsEntry = ttk.Entry(master=frame2, width=50)
predictButton = tk.Button(text="Predict Rating", width=15, command=predictRating,
                          background="#f0ca0e", foreground="red", font=("Cooper Black", 15))
window.rowconfigure(index=2, weight=1)
window.rowconfigure(index=3, weight=1)
window.columnconfigure(index=0, weight=1)
window.columnconfigure(index=1, weight=1)
frame1.columnconfigure(index=0, pad=5)
frame1.columnconfigure(index=1, pad=5)
header.grid(row=0, column=0, columnspan=2, pady=5)
instructions.grid(row=1, column=0, columnspan=2)
titleLabel.grid(row=0, column=0, pady=5, sticky="e")
titleEntry.grid(row=0, column=1, columnspan=5, pady=5, sticky="w")
contentRatingLabel.grid(row=1, column=0, pady=5, sticky="e")
gRating.grid(row=1, column=1, pady=5)
pgRating.grid(row=1, column=2, pady=5)
pg13Rating.grid(row=1, column=3, pady=5)
rRating.grid(row=1, column=4, pady=5)
nrRating.grid(row=1, column=5, pady=5)
originalReleaseDateLabel.grid(row=2, column=0, pady=5, sticky="e")
originalReleaseDateEntry.grid(row=2, column=1, columnspan=5, pady=5, sticky="w")
streamingReleaseDateLabel.grid(row=3, column=0, pady=5, sticky="e")
streamingReleaseDateEntry.grid(row=3, column=1, columnspan=5, pady=5, sticky="w")
runtimeLabel.grid(row=4, column=0, pady=5, sticky="e")
runtimeEntry.grid(row=4, column=1, columnspan=5, pady=5, sticky="w")
productionCoLabel.grid(row=0, column=0, pady=5, sticky="e")
productionCoEntry.grid(row=0, column=1, pady=5, sticky="w")
genresLabel.grid(row=1, column=0, pady=5, sticky="e")
genresEntry.grid(row=1, column=1, pady=5, sticky="w")
directorsLabel.grid(row=2, column=0, pady=5, sticky="e")
directorsEntry.grid(row=2, column=1, pady=5, sticky="w")
authorsLabel.grid(row=3, column=0, pady=5, sticky="e")
authorsEntry.grid(row=3, column=1, pady=5, sticky="w")
actorsLabel.grid(row=4, column=0, pady=5, sticky="e")
actorsEntry.grid(row=4, column=1, pady=5, sticky="w")
frame1.grid(row=2, column=0)
frame2.grid(row=2, column=1)
predictButton.grid(row=3, column=0, columnspan=2)
window.mainloop()
