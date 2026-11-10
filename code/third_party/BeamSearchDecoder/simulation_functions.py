# This file is part of BeamSearchDecoder.

# Copyright (c) 2025 IonQ, Inc., all rights reserved

# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License (CC BY-NC-SA 4.0).

# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc-sa/4.0/


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import sinter
import stim
import multiprocessing
import time
from beamsearch import BeamSearch
from sinter_beamsearch import SinterDecoder_BeamSearch
from stimbposd import BPOSD, SinterDecoder_BPOSD

decoder_dictionary = {
    "beam8_230iters": SinterDecoder_BeamSearch(),
    "beam32_340iters": SinterDecoder_BeamSearch(beam_width=32, initial_iters=40, iters_per_round=30),
    "beam64_640iters": SinterDecoder_BeamSearch(max_rounds=20, beam_width=64, initial_iters=40, iters_per_round=30),
    "beam64_32res_640iters": SinterDecoder_BeamSearch(
        max_rounds=20, beam_width=64, num_results=32, initial_iters=40, iters_per_round=30
    ),
    "bp30+osd": SinterDecoder_BPOSD(max_bp_iters=30, bp_method="min_sum", osd_order=10, osd_method="osd_cs")
}

distance_dictionary = {
    (144, 12): 12,
    (90, 8): 10,
    (450, 32): 8
}

StimCircuit_dictionary = {
    (144, 12, 0.001, "x"): "BB[[144,12,12]],memory_X,error_rate=0.001,syndrome_rounds=12.stim",
    (144, 12, 0.001, "z"): "BB[[144,12,12]],memory_Z,error_rate=0.001,syndrome_rounds=12.stim",
    (144, 12, 0.002, "x"): "BB[[144,12,12]],memory_X,error_rate=0.002,syndrome_rounds=12.stim",
    (144, 12, 0.002, "z"): "BB[[144,12,12]],memory_Z,error_rate=0.002,syndrome_rounds=12.stim",
    (144, 12, 0.003, "x"): "BB[[144,12,12]],memory_X,error_rate=0.003,syndrome_rounds=12.stim",
    (144, 12, 0.003, "z"): "BB[[144,12,12]],memory_Z,error_rate=0.003,syndrome_rounds=12.stim",
    (144, 12, 0.004, "x"): "BB[[144,12,12]],memory_X,error_rate=0.004,syndrome_rounds=12.stim",
    (144, 12, 0.004, "z"): "BB[[144,12,12]],memory_Z,error_rate=0.004,syndrome_rounds=12.stim",
    (144, 12, 0.005, "x"): "BB[[144,12,12]],memory_X,error_rate=0.005,syndrome_rounds=12.stim",
    (144, 12, 0.005, "z"): "BB[[144,12,12]],memory_Z,error_rate=0.005,syndrome_rounds=12.stim",
    (144, 12, 0.006, "x"): "BB[[144,12,12]],memory_X,error_rate=0.006,syndrome_rounds=12.stim",
    (144, 12, 0.006, "z"): "BB[[144,12,12]],memory_Z,error_rate=0.006,syndrome_rounds=12.stim",
    (90, 8, 0.002, "x"): "BB[[90,8,10]],memory_X,error_rate=0.002,syndrome_rounds=10.stim",
    (90, 8, 0.002, "z"): "BB[[90,8,10]],memory_Z,error_rate=0.002,syndrome_rounds=10.stim",
    (90, 8, 0.003, "x"): "BB[[90,8,10]],memory_X,error_rate=0.003,syndrome_rounds=10.stim",
    (90, 8, 0.003, "z"): "BB[[90,8,10]],memory_Z,error_rate=0.003,syndrome_rounds=10.stim",
    (90, 8, 0.004, "x"): "BB[[90,8,10]],memory_X,error_rate=0.004,syndrome_rounds=10.stim",
    (90, 8, 0.004, "z"): "BB[[90,8,10]],memory_Z,error_rate=0.004,syndrome_rounds=10.stim",
    (90, 8, 0.005, "x"): "BB[[90,8,10]],memory_X,error_rate=0.005,syndrome_rounds=10.stim",
    (90, 8, 0.005, "z"): "BB[[90,8,10]],memory_Z,error_rate=0.005,syndrome_rounds=10.stim",
    (90, 8, 0.002, "xyz_x"): "BB[[90,8,10]],memory_X,error_rate=0.002,syndrome_rounds=10,XYZ_circuit.stim",
    (90, 8, 0.002, "xyz_z"): "BB[[90,8,10]],memory_Z,error_rate=0.002,syndrome_rounds=10,XYZ_circuit.stim",
    (90, 8, 0.003, "xyz_x"): "BB[[90,8,10]],memory_X,error_rate=0.003,syndrome_rounds=10,XYZ_circuit.stim",
    (90, 8, 0.003, "xyz_z"): "BB[[90,8,10]],memory_Z,error_rate=0.003,syndrome_rounds=10,XYZ_circuit.stim",
    (90, 8, 0.004, "xyz_x"): "BB[[90,8,10]],memory_X,error_rate=0.004,syndrome_rounds=10,XYZ_circuit.stim",
    (90, 8, 0.004, "xyz_z"): "BB[[90,8,10]],memory_Z,error_rate=0.004,syndrome_rounds=10,XYZ_circuit.stim",
    (90, 8, 0.005, "xyz_x"): "BB[[90,8,10]],memory_X,error_rate=0.005,syndrome_rounds=10,XYZ_circuit.stim",
    (90, 8, 0.005, "xyz_z"): "BB[[90,8,10]],memory_Z,error_rate=0.005,syndrome_rounds=10,XYZ_circuit.stim",
    (450, 32, 0.002, "x"): "HGP[[450,32,8]],memory_X,error_rate=0.002,syndrome_rounds=8.stim",
    (450, 32, 0.002, "z"): "HGP[[450,32,8]],memory_Z,error_rate=0.002,syndrome_rounds=8.stim",
    (450, 32, 0.003, "x"): "HGP[[450,32,8]],memory_X,error_rate=0.003,syndrome_rounds=8.stim",
    (450, 32, 0.003, "z"): "HGP[[450,32,8]],memory_Z,error_rate=0.003,syndrome_rounds=8.stim",
    (450, 32, 0.004, "x"): "HGP[[450,32,8]],memory_X,error_rate=0.004,syndrome_rounds=8.stim",
    (450, 32, 0.004, "z"): "HGP[[450,32,8]],memory_Z,error_rate=0.004,syndrome_rounds=8.stim",
    (450, 32, 0.005, "x"): "HGP[[450,32,8]],memory_X,error_rate=0.005,syndrome_rounds=8.stim",
    (450, 32, 0.005, "z"): "HGP[[450,32,8]],memory_Z,error_rate=0.005,syndrome_rounds=8.stim"
}

def one_point_simulation(
    n: int,
    k: int,
    p_CNOT: float,
    circuit_type: str,
    decoder: str,
    maximum_shots: int = 100_000,
    maximum_errors: int = 100,
):
    if circuit_type == "normal":
        if (n, k, p_CNOT, "x") in StimCircuit_dictionary:
            x_circuit = stim.Circuit.from_file("StimCircuit/" + StimCircuit_dictionary[(n, k, p_CNOT, "x")])
        else:
            raise ValueError(f"({n}, {k}, {p_CNOT}, {circuit_type}) circuit not available")
        if (n, k, p_CNOT, "z") in StimCircuit_dictionary:
            z_circuit = stim.Circuit.from_file("StimCircuit/" + StimCircuit_dictionary[(n, k, p_CNOT, "z")])
        else:
            raise ValueError(f"({n}, {k}, {p_CNOT}, {circuit_type}) circuit not available")
    elif circuit_type == "XYZ":
        if (n, k, p_CNOT, "xyz_x") in StimCircuit_dictionary:
            x_circuit = stim.Circuit.from_file("StimCircuit/" + StimCircuit_dictionary[(n, k, p_CNOT, "xyz_x")])
        else:
            raise ValueError(f"({n}, {k}, {p_CNOT}, {circuit_type}) circuit not available")
        if (n, k, p_CNOT, "xyz_z") in StimCircuit_dictionary:
            z_circuit = stim.Circuit.from_file("StimCircuit/" + StimCircuit_dictionary[(n, k, p_CNOT, "xyz_z")])
        else:
            raise ValueError(f"({n}, {k}, {p_CNOT}, {circuit_type}) circuit not available")
    else:
        raise ValueError("circuit_type should be either normal or XYZ")

    x_samples = sinter.collect(
        num_workers=multiprocessing.cpu_count() - 1,
        max_shots=maximum_shots,
        max_errors=maximum_errors,
        tasks=[sinter.Task(circuit=x_circuit, json_metadata={"p_CNOT": p_CNOT, "x_or_z": "x"})],
        decoders=[decoder],
        custom_decoders=decoder_dictionary,
        print_progress=True,
    )
    x_num_errors = x_samples[0].errors
    x_num_shots = x_samples[0].shots
    x_logical_error_rate = (x_num_errors / x_num_shots) / distance_dictionary[(n, k)]
    
    z_samples = sinter.collect(
        num_workers=multiprocessing.cpu_count() - 1,
        max_shots=maximum_shots,
        max_errors=maximum_errors,
        tasks=[sinter.Task(circuit=z_circuit, json_metadata={"p_CNOT": p_CNOT, "x_or_z": "z"})],
        decoders=[decoder],
        custom_decoders=decoder_dictionary,
        print_progress=True,
    )
    z_num_errors = z_samples[0].errors
    z_num_shots = z_samples[0].shots
    z_logical_error_rate = (z_num_errors / z_num_shots) / distance_dictionary[(n, k)]

    tt_logical_error_rate = x_logical_error_rate + z_logical_error_rate

    print(f"{decoder} {circuit_type} circuit memory_X results:")
    print(f"num errors is {x_num_errors}, num shots is {x_num_shots}, X logical error rate is {x_logical_error_rate}")
    print(f"{decoder} {circuit_type} circuit memory_Z results:")
    print(f"num errors is {z_num_errors}, num shots is {z_num_shots}, Z logical error rate is {z_logical_error_rate}")
    print(f"Total (X+Z) logical error rate is {tt_logical_error_rate}")
    # print(sinter.CSV_HEADER.strip())
    # for sample in x_samples:
    #     print(sample.to_csv_line().strip())
    # for sample in z_samples:
    #     print(sample.to_csv_line().strip())


def decoding_time(p_CNOT: float, decoder: str, num_shots: int = 10_000):
    print(f"{decoder} decodes [[144,12,12]] BB code with 12 syndrome extraction rounds at noise rate {p_CNOT}:")
    if p_CNOT == 0.0005:
        z_circuit = stim.Circuit.from_file("StimCircuit/BB[[144,12,12]],memory_Z,error_rate=0.0005,syndrome_rounds=12.stim")
    elif p_CNOT == 0.001:
        z_circuit = stim.Circuit.from_file("StimCircuit/BB[[144,12,12]],memory_Z,error_rate=0.001,syndrome_rounds=12.stim")
    else:
        raise ValueError("Only support p_CNOT = 0.0005 or 0.001")
    sampler = z_circuit.compile_detector_sampler()
    shots, observables = sampler.sample(num_shots, separate_observables=True)
    if decoder == "bp30+osd":
        mydec = BPOSD(
            z_circuit.detector_error_model(),
            max_bp_iters=30,
            bp_method="min_sum",
            osd_order=10,
            osd_method="osd_cs",
        )
    elif decoder == "beam8_230iters":
        mydec = BeamSearch(z_circuit.detector_error_model())
    elif decoder == "beam32_340iters":
        mydec = BeamSearch(
            z_circuit.detector_error_model(), beam_width=32, initial_iters=40, iters_per_round=30
        )
    elif decoder == "beam64_640iters":
        mydec = BeamSearch(
            z_circuit.detector_error_model(),
            max_rounds=20,
            beam_width=64,
            initial_iters=40,
            iters_per_round=30,
        )
    else:
        raise ValueError("Only support 4 decoders bp30+osd, beam8_230iters, beam32_340iters, beam64_640iters")

    time_array = np.zeros(num_shots)
    for i in range(num_shots):
        start_time = time.perf_counter()
        mydec.decode(shots[i])
        end_time = time.perf_counter()
        time_array[i] = end_time - start_time
    time_array.sort()
    tail_index = int(0.999 * num_shots) - 1
    print(f"The average decoding time is {time_array.mean()}, and the 99.9 percentile decoding time is {time_array[tail_index]}")


def generate_tasks(n: int, k: int):
    if n == 144 and k == 12:
        physical_error_rate_table = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
        code_name = "[[144, 12, 12]] BB code"
    elif n == 90 and k == 8:
        physical_error_rate_table = [0.002, 0.003, 0.004, 0.005]
        code_name = "[[90, 8, 10]] BB code"
    elif n == 450 and k == 32:
        physical_error_rate_table = [0.002, 0.003, 0.004, 0.005]
        code_name = "[[450, 32, 8]] HGP code"
    else:
        raise ValueError(f"Do not support this (n={n}, k={k}) pair")
    for p_CNOT in physical_error_rate_table:
        for x_or_z in ["x", "z"]:
            yield sinter.Task(
                circuit=stim.Circuit.from_file("StimCircuit/" + StimCircuit_dictionary[(n, k, p_CNOT, x_or_z)]),
                json_metadata={
                    "code": code_name,
                    "syndrome_rounds": distance_dictionary[(n, k)],
                    "p_CNOT": p_CNOT,
                    "x_or_z": x_or_z,
                },
            )


def generate_tasks_for_XYZ_circuit():
    physical_error_rate_table = [0.002, 0.003, 0.004, 0.005]
    code_name = "[[90, 8, 10]] BB code"
    for p_CNOT in physical_error_rate_table:
        for x_or_z in ["xyz_x", "xyz_z"]:
            yield sinter.Task(
                circuit=stim.Circuit.from_file("StimCircuit/" + StimCircuit_dictionary[(90, 8, p_CNOT, x_or_z)]),
                json_metadata={
                    "code": code_name,
                    "syndrome_rounds": distance_dictionary[(90, 8)],
                    "p_CNOT": p_CNOT,
                    "x_or_z": x_or_z,
                },
            )


def full_simulation(n: int, k: int, maximum_shots: int = 1_000_000, maximum_errors: int = 100):
    if n == 144 and k == 12:
        file_name = "[[144,12,12]]_BB_code.csv"
        decoder_list = ['bp30+osd', 'beam8_230iters', 'beam32_340iters', 'beam64_640iters', 'beam64_32res_640iters']
    elif n == 90 and k == 8:
        file_name = "[[90,8,10]]_BB_code.csv"
        decoder_list = ['bp30+osd', 'beam8_230iters', 'beam64_640iters']
    elif n == 450 and k == 32:
        file_name = "[[450,32,8]]_HGP_code.csv"
        decoder_list = ['bp30+osd', 'beam8_230iters', 'beam64_640iters']
    else:
        raise ValueError(f"Do not support this (n={n}, k={k}) pair")
    temp_file = "simulation_results/temp_files/" + file_name
    final_file = f"simulation_results/" + file_name
    num_cpus = multiprocessing.cpu_count() - 1
    samples = sinter.collect(
        num_workers=num_cpus,
        max_shots=maximum_shots,
        max_errors=maximum_errors,
        tasks=generate_tasks(n, k),
        decoders=decoder_list,
        custom_decoders=decoder_dictionary,
        save_resume_filepath=temp_file,
        # BB_code.csv stores the (intermediate) results produced by different workers/CPUs.
        # If the python interpreter is stopped or killed, calling this method again
        # with the same save_resume_filepath will load the previous results
        # from the file so it can resume where it left off.
        print_progress=True,
    )

    # Save results to final_results.csv
    with open(final_file, "w") as csvfile:
        print(sinter.CSV_HEADER.strip(), file=csvfile)
        for sample in samples:
            print(sample.to_csv_line().strip(), file=csvfile)
    
    if n == 90 and k == 8:
        file_name = "[[90,8,10]]_BB_code_XYZ_circuit.csv"
        decoder_list = ['bp30+osd', 'beam64_640iters']
        temp_file = "simulation_results/temp_files/" + file_name
        final_file = f"simulation_results/" + file_name
        num_cpus = multiprocessing.cpu_count() - 1
        samples = sinter.collect(
            num_workers=num_cpus,
            max_shots=maximum_shots,
            max_errors=maximum_errors,
            tasks=generate_tasks_for_XYZ_circuit(),
            decoders=decoder_list,
            custom_decoders=decoder_dictionary,
            save_resume_filepath=temp_file,
            # BB_code.csv stores the (intermediate) results produced by different workers/CPUs.
            # If the python interpreter is stopped or killed, calling this method again
            # with the same save_resume_filepath will load the previous results
            # from the file so it can resume where it left off.
            print_progress=True,
        )

        # Save results to final_results.csv
        with open(final_file, "w") as csvfile:
            print(sinter.CSV_HEADER.strip(), file=csvfile)
            for sample in samples:
                print(sample.to_csv_line().strip(), file=csvfile)


def plot_logical_error_rate(n: int, k: int):
    if n == 144 and k == 12:
        physical_error_rate_table = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
        code_name = "[[144, 12, 12]] BB code"
        file_name = "simulation_results/[[144,12,12]]_BB_code.csv"
        decoder_list = ['bp30+osd', 'beam8_230iters', 'beam32_340iters', 'beam64_640iters', 'beam64_32res_640iters']
    elif n == 90 and k == 8:
        physical_error_rate_table = [0.002, 0.003, 0.004, 0.005]
        code_name = "[[90, 8, 10]] BB code"
        file_name = "simulation_results/[[90,8,10]]_BB_code.csv"
        decoder_list = ['bp30+osd', 'beam8_230iters', 'beam64_640iters']
    elif n == 450 and k == 32:
        physical_error_rate_table = [0.002, 0.003, 0.004, 0.005]
        code_name = "[[450, 32, 8]] HGP code"
        file_name = "simulation_results/[[450,32,8]]_HGP_code.csv"
        decoder_list = ['bp30+osd', 'beam8_230iters', 'beam64_640iters']
    else:
        raise ValueError(f"Do not support this (n={n}, k={k}) pair")
    
    error_rate_dict = {item: index for index, item in enumerate(physical_error_rate_table)}
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
    df["json_metadata"] = df["json_metadata"].apply(lambda x: json.loads(x) if pd.notna(x) and x != "" else {})
    df.columns = df.columns.str.strip()

    for decoder in decoder_list:
        logical_error_list = [0.0 for _ in physical_error_rate_table]
        for i in range(df.shape[0]):
            if df["decoder"][i] == decoder:
                if df["json_metadata"][i]["p_CNOT"] in error_rate_dict:
                    logical_error_list[error_rate_dict[df["json_metadata"][i]["p_CNOT"]]] += df["errors"][i] / df["shots"][i]
        logical_error_list = [item / distance_dictionary[(n, k)] for item in logical_error_list]
        # print(decoder)
        # print(physical_error_rate_table)
        # print(logical_error_list)
        plt.plot(
            physical_error_rate_table,
            logical_error_list,
            marker="o",
            label=decoder,
        )
    
    if n == 90 and k == 8:
        physical_error_rate_table = [0.002, 0.003, 0.004, 0.005]
        file_name = "simulation_results/[[90,8,10]]_BB_code_XYZ_circuit.csv"
        decoder_list = ['bp30+osd', 'beam64_640iters']
        error_rate_dict = {item: index for index, item in enumerate(physical_error_rate_table)}
        try:
            df = pd.read_csv(file_name)
        except FileNotFoundError:
            print(f"Error: The file '{file_name}' was not found.")
        df["json_metadata"] = df["json_metadata"].apply(lambda x: json.loads(x) if pd.notna(x) and x != "" else {})
        df.columns = df.columns.str.strip()

        for decoder in decoder_list:
            logical_error_list = [0.0 for _ in physical_error_rate_table]
            for i in range(df.shape[0]):
                if df["decoder"][i] == decoder:
                    if df["json_metadata"][i]["p_CNOT"] in error_rate_dict:
                        logical_error_list[error_rate_dict[df["json_metadata"][i]["p_CNOT"]]] += df["errors"][i] / df["shots"][i]
            logical_error_list = [item / distance_dictionary[(n, k)] for item in logical_error_list]
            # print(decoder + "_XYZ")
            # print(physical_error_rate_table)
            # print(logical_error_list)
            plt.plot(
                physical_error_rate_table,
                logical_error_list,
                marker="o",
                label=decoder+"_XYZ",
            )
    
    plt.loglog()
    plt.xlim([8e-4, 7e-3])
    plt.xlabel("physical error rate")
    plt.ylabel("logical error rate per syndrome extraction round")
    plt.legend()
    plt.title(code_name)
    plt.grid(which="major")
    plt.grid(which="minor")
    plt.show()