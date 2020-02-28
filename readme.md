# TrafficLightRL

This is a new version of a reinforcement learning pipeline for intelligent traffic control. Usage and more information can be found below.

## Usage

Start an experiment:

``python -O run_batch.py``

Here, ``-O`` option cannot be omitted unless debug is necessary. In the file ``run_batch.py``, the name of memo can be changed. There are four modules executed sequentially:

* ``runexp.py``

  Run the pipeline under different traffic flows. Specific traffic flow files as well as basic configuration can be assigned in this file. For details about config, please turn to ``config.py``. 

* ``testexp.py``

  With models trained after different rounds, we can test them using this module. ``run_cnts`` can be adjusted.

* ``summary_detail.py``

  Calculate durations with our models as control agent for traffic light in the train and test mode and write results to ``train_results.csv`` and  ``test_results.csv`` respecitvely. At the same time, add summaries of this experiment setting to ``total_train_results.csv`` and ``total_test_results.csv``.

* ``summary_plot.py``

  Plot figures of ``Round V.S Duration`` with obtained results.

For most cases, you might only modify ``memo`` in ``run_batch.py``, traffic files and config parameters in ``runexp.py``.

## Agent

* ``agent.py``

  A abstract class of different agents.

* ``./baseline/fixedtime_agent.py``

  An agent to implement fixedtime control on traffic lights.

* ``network_agent.py``

  A abstract class of neural network based agents.  All methods are defined in this file but ``build_network()``, which means only ``build_network()`` is necessary in specific network agents.

* ``./baseline/deeplight_agent.py``

  A DQN agent with phase selecor.

* ``simple_dqn_agent.py``

  A simple DQN agent

## Others

More details about this project are demonstrated in this part.

* ``config.py`` 

  The whole configuration of this project. Note that some parameters will be replaced in ``runexp.py`` while others can only be changed in this file, please be very careful!!!

* ``pipeline.py``

  The whole pipeline is implemented in this module:

  Start a SUMO environment, run a simulation for certain time(one round), construct samples from raw log data, update the model and model pooling.

* ``generator.py``

  A generator to load a model, start a SUMO enviroment, conduct a simulation and log the results.

* ``sumo_env.py``

  Define a SUMO environment to interact with SUMO and obtain needed data like features.

* ``construct_sample.py``

* Construct training samples from original data. Select desired state features in the config and compute the corrsponding average/instant reward with specific measure time.

* ``updater.py``

  Define a class of updater for model updating.
  
 ## baseline
 to run some baseline algorithms (formula, fixedtime, lit (deeplight)):
 * ``python -O run_baseline_batch.py``
