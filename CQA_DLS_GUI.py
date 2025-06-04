#This python script contains a GUI that can be used to calculate geometic metrics of RTstructs.
#Ideally it is not required to fix anything in this file (if the setup is the same as proposed by Van Acht et al.)
#Your own input is required at the beginning of the MyGUI class

# DISCLAIMER:
#
# This code is provided "as-is".
# The author is not responsible for any errors, damages, or consequences that arise from the use of
# this code. It is the user's responsibility to thoroughly validate and test the code before using it
# in any medical or clinical environment. Ensure that all necessary precautions are taken and that
# the code complies with all applicable regulations and standards.
#
# Use at your own risk.

#Carefully read the Read Me to ensure save and more easy employement of the script

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog
from calculator_functions import *
from GUI_functions import *


if __name__ == "__main__":
    import sys

    # Call the function based on the terminal argument
    if len(sys.argv) > 1 and sys.argv[1] == "get_scores":
        start_date = input("Enter the start date (DD-MM-YYYY): ")
        end_date = input("Enter the end date (DD-MM-YYYY): ")
        get_scores(start_date=start_date,end_date=end_date)

class MyGUI():
    def __init__(self, root):
        #PERSONAL INPUT, INSTITUTE SPECIFIC
        self.data_path =  r"L:\GFR\R2401_Hurkmans Ratified PhD"
        self.save_path = r"L:\GFR\TPS_RS\Innovatie en Onderzoek\Niels\BQA"
        self.control_limits_file = r"L:\GFR\TPS_RS\Innovatie en Onderzoek\Niels\BQA\control_limits.json"
        self.exclusion_path = 'Exclusion_ROIs.csv'
        self.N_min = 10
        self.xdim, self.ydim, self.zdim = 1.17, 1.17, 3.0
        self.value1,self.value2,self.value3 = 2.5, 9 , 6
 
        #START OF GUI
        self.root = root
        self.root.title("Continuous QA - DLS")

        # Associate the close event with the handler method
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Create input labels and entry widgets
        self.label1 = tk.Label(root, text="Start date:")
        self.entry1 = tk.Entry(root)
        self.entry1.insert(0, "28-08-2024")  # Set default value for Start date
        
        self.label2 = tk.Label(root, text="End date:")
        self.entry2 = tk.Entry(root)
        current_date = datetime.now().strftime("%d-%m-%Y")  # Get current date in DD-MM-YYYY format
        self.entry2.insert(0, current_date)  # Set default value for Eind date
        
        # Create the "Krijg data" button
        self.button = tk.Button(root, text="Get data", command=self.get_data)
        
        # Create a label for status messages
        self.status_label = tk.Label(root, text="")
        
        # Arrange widgets using grid layout
        self.label1.grid(row=0, column=0, padx=1, pady=1)
        self.entry1.grid(row=0, column=1, padx=1, pady=1)
        
        self.label2.grid(row=1, column=0, padx=1, pady=1)
        self.entry2.grid(row=1, column=1, padx=1, pady=1)
        
        self.button.grid(row=2, columnspan=2, padx=1, pady=1)
        self.status_label.grid(row=3, columnspan=2, padx=1, pady=1)

        # Create a label for the ROI selection
        self.roi_label = tk.Label(self.root, text="Select ROIs")

        # Create a list to store selected ROIs
        self.selected_rois = []
        # Create a Listbox for displaying ROIs
        self.roi_listbox = tk.Listbox(self.root, selectmode=tk.MULTIPLE)
        self.rois= get_roi_list(self.data_path)
        for roi in self.rois:
            self.roi_listbox.insert(tk.END, roi)

        # Arrange widgets using grid layout
        # (Place the Listbox and label wherever you want them in your layout)
        self.roi_label.grid(row=4, column=0, columnspan=2, padx=1, pady=1)
        self.roi_listbox.grid(row=5, column=0, columnspan=2, rowspan=12, padx=1, pady=1)

        # Create a button to get selected ROIs
        self.get_rois_button = tk.Button(self.root, text="Plot selected ROIs", command=self.plot_selected_rois)
        self.get_rois_button.grid(row=17,column=0,columnspan=2,padx=1,pady=1)

        #initialize empty figure
        self.fig = Figure(figsize=(8, 6))

         # Create a canvas for the plot
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.plot_canvas.get_tk_widget().grid(row=0, column=2, rowspan=40, padx=1, pady=1)

        self.save_button = tk.Button(self.root, text="Save plot", command=self.save_figure)
        self.save_button.grid(row=18, column=0, columnspan =2, padx=1, pady=1)

        self.excel_button = tk.Button(self.root, text="Create Excel", command=self.create_excel_file)
        self.excel_button.grid(row=19, column=0, columnspan =2, padx=1, pady=1)

        self.trend_button = tk.Button(self.root, text="Nelson Analysis", command=self.nelson_analysis)
        self.trend_button.grid(row=20, column=0, columnspan =2, padx=1, pady=1)


    def on_close(self):
        # Handle the window closing event
        self.root.destroy()
    
    def get_data(self):
        self.button.config(state=tk.DISABLED)  # Disable the button initially
        # For demonstration purposes, let's simulate a delay
        self.status_label.config(text="Scoring new patients...")
        self.root.after(200,self.show_scores)  # Simulate a delay of 0.2 seconds
    
    def show_scores(self):
        input1 = self.entry1.get()
        input2 = self.entry2.get()
        self.message = get_scores(self.data_path,input1,input2,self.xdim,self.ydim,self.zdim,self.exclusion_path)
        # For now, just display a message
        self.status_label.config(text=self.message)
        self.button.config(state=tk.NORMAL)  # Enable the button after scores are displayed


    def plot_selected_rois(self):
        selected_indices = self.roi_listbox.curselection()
        rois_to_plot = [self.rois[i] for i in selected_indices]
        input1 = self.entry1.get()
        input2 = self.entry2.get()
        
        plt.ioff()  # Disable interactive mode

        # Generate the plot (ensure this function returns a fully populated figure)
        new_fig = make_plot(rois_to_plot, input1, input2,self.data_path)

        # Ensure the figure contains all content
        self.fig = new_fig

        # Render the canvas to ensure it's ready before saving
        self.plot_canvas.draw()

        # Update the existing canvas with the new figure
        self.plot_canvas.get_tk_widget().destroy()
        self.plot_canvas = FigureCanvasTkAgg(new_fig, master=self.root)
        self.plot_canvas.get_tk_widget().grid(row=1, column=2, rowspan=20, padx=1, pady=1)

    def create_excel_file(self):
        selected_indices = self.roi_listbox.curselection()
        rois_to_plot = [self.rois[i] for i in selected_indices]
        input1 = self.entry1.get()
        input2 = self.entry2.get()
        make_excel_file(rois_to_plot,input1,input2,self.data_path,self.save_path)

    def save_figure(self):
        # Check if self.fig is valid
        if self.fig is None:
            print("No figure available to save!")
            return

        # Open the file save dialog
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png"),
                                                        ("All files", "*.*")])

        if file_path:
            try:
                # Ensure the figure is fully drawn
                self.plot_canvas.draw()

                # Now save the figure
                self.fig.savefig(file_path)

                print(f"Figure saved as {file_path}")
            except Exception as e:
                print(f"Error saving figure: {e}")
        else:
            print("Save operation canceled.")  # Handle cancellation

    def nelson_analysis(self):
        # Load dictionary from the JSON file
        with open(self.control_limits_file , 'r') as file:
            control_limits = json.load(file)

        Rule1, Rule2,Rule3 = nelson_detection(data_path=self.data_path,limits=control_limits,start_date=20240828,value1=self.value1,value2=self.value2,value3=self.value3,N_min=self.N_min)

        csv_path1 = os.path.join(self.save_path,'Nelson_Rule_1.csv')  # Replace with the actual path to your CSV file
        update_csv_1(csv_path1, Rule1)

        csv_path2 = os.path.join(self.save_path,'Nelson_Rule_2.csv')  # Replace with the actual path to your CSV file
        update_csv_23(csv_path2, Rule2)

        csv_path3 = os.path.join(self.save_path,'Nelson_Rule_3.csv')  # Replace with the actual path to your CSV file
        update_csv_23(csv_path3, Rule3)

if __name__ == "__main__":
    root = tk.Tk()
    app = MyGUI(root)
    root.mainloop()
