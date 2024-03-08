import os
import sys
import subprocess
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

if "sund" not in os.listdir('./custom_package'):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target=./custom_package", 'https://isbgroup.eu/edu/assets/sund-1.0.1.tar.gz#sha256=669a1d05c5c8b68500086e183d831650277012b3ea57e94356de1987b6e94e3e'])

sys.path.append('./custom_package')
import sund


# Setup the model and simulation
model_name = 'M2'
sund.installModel(f"{model_name}.txt")
model_class = sund.importModel(model_name)
model = model_class() 
model.parametervalues = [486.35305579074378,4030.1954977108358,203.67073905125929,99999.9250766247,372.86765513472358]# = [k1, k2, kfeed, k4, k5]


features = model.featurenames[:-1] # Remove the input from the list

# Start the app
st.title("Prediction data generator")
st.markdown(""" 
        This app allow you to generate data for a given prediction of your choice. Define the inputs corresponding to your prediction, i.e. how the activator is changed over time. And define which time points to you want to get the prediction for.

""")

st.markdown("""
            ## Define the activation
            Specify the time points and the dose strength of the activator in the same time points. Separate multiple values with comma.
            """)
tvalues_input = st.text_input('Time points (separate with comma):', value = "0, 5")
tvalues = [float(t.strip()) for t in tvalues_input.split(',')]

fvalues_input = st.text_input('Dose strengths (separate with comma):', value = "0, 1")
fvalues = [float(f.strip()) for f in fvalues_input.split(',')]

if tvalues[0] == 0:
    ic = model.statevalues
    ic[model.statenames.index('A')] = fvalues[0]

# Plot the input
show_plot = st.toggle('Visualize the defined input?')
if show_plot:
    input_activity = sund.Activity(timeunit='m')
    input_activity.AddOutput(sund.PIECEWISE_CONSTANT, "A_in", tvalues = tvalues, fvalues = [0] + fvalues)

    sim = sund.Simulation(models = [model], activities=[input_activity], timeunit='m')
    sim.Simulate(timevector = np.linspace(0.0, max(tvalues)+5,500), resetstatesderivatives=True)
    sim_results = pd.DataFrame(sim.featuredata,columns=sim.featurenames)
    sim_results.insert(0, 'Time', sim.timevector)


    st.subheader("Visualization of the defined activation")
    st.line_chart(sim_results, x="Time", y="input")

# Define when to measure

st.markdown("""
            ## Define the time points to measure in

            Specify the time points to measure in, separate the values with a comma.
            """)
time_input = st.text_input('Time points to measure in (separate with comma):',
                            value = "0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60")

times = [float(t.strip()) for t in time_input.split(',')]
times = sorted(times)
feature = st.selectbox("Component of the system to measure? ", features, index=1)


# Setup the simulation, with resetting of the input between time points

# Insert 0 into fvalues between two consecutive time points with nonzero values to be able to trigger the event multiple times
tvalues_expanded = [tvalues[0]]
fvalues_expanded = [fvalues[0]]

for idx in range(1, len(tvalues)):
    if fvalues[idx-1] !=0 and fvalues[idx] != 0:
        tvalues_expanded.append((tvalues[idx-1]+tvalues[idx])/2) # insert the middle point
        fvalues_expanded.append(0)
    tvalues_expanded.append(tvalues[idx])
    fvalues_expanded.append(fvalues[idx])

activity = sund.Activity(timeunit='m')
activity.AddOutput(sund.PIECEWISE_CONSTANT, "A_in", 
                   tvalues = tvalues_expanded, fvalues = [0] + fvalues_expanded)
sim = sund.Simulation(models = [model], activities=[activity], timeunit='m')

# Simulate the new experiment
end_time = max(times)+5
zero_not_in_times = times[0]>0

if zero_not_in_times:
    times = [0] + times

sim.Simulate(timevector = np.linspace(times[0], times[-1],500), resetstatesderivatives=True) # highres to debugging purposes
sim_results = pd.DataFrame(sim.featuredata,columns=sim.featurenames)
sim_results.insert(0, 'Time', sim.timevector)

# Simulate the new experiment mean and SEM
sim.Simulate(timevector = times, resetstatesderivatives=True)

mean = sim.featuredata[:,sim.featurenames.index(feature)]
mean = mean+np.random.normal(0, 0.004, len(mean))
mean = list(np.abs(mean))
if zero_not_in_times:
    mean.pop(0)
    times.pop(0)
sem = list(np.random.uniform(0.003, 0.012, len(mean)))
st.markdown("## The experimental data for the defined prediction experiment")
st.subheader("The new data as a time-series plot")

layout = go.Layout(
    margin=go.layout.Margin(
        l=0, # left margin
        r=0, # right margin
        b=0, # bottom margin
        t=0 # top margin
    )
)

fig = go.Figure(layout=layout)
fig.add_trace(go.Scatter(x=sim_results['Time'], y=sim_results[feature], mode='lines', name=feature))
fig.add_trace(go.Scatter(x=times, y=mean, mode='markers', name="Experimental data", error_y=dict(type='data', array=sem, visible=True)))
fig.update_layout(showlegend=False, xaxis_title="Time (min)", yaxis_title=feature)

st.plotly_chart(fig)

# Get experimental data from simulation asked for
data = {"input": {"t": tvalues, "f": fvalues},
        "time": times,
        "mean": mean,
        "SEM": sem}

st.subheader("The new data as a table")
data_table = data.copy()
data_table.pop("input")
st.table(pd.DataFrame(data_table))

st.markdown(f"""
### The new data in the JSON format 
```json
{json.dumps({"prediction": data}, indent=4)}
```
""")

