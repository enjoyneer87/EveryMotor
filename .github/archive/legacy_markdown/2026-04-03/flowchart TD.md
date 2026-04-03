flowchart TD
    %% 1. Raw Data Input
    subgraph Data Source
        H5[PyMCAD H5 Files]
    end

    %% 2. Data Processing Pipeline
    subgraph Phase 2: doe_data_utils.py 
        direction TB
        Scatter[scatter_elem_to_node]
        
        %% GINO Logic
        Scatter --> BuildG[build_gino_sample]
        BuildG --> |"Inputs: 7 Channels\nTargets: 4 Channels (Bx, By, A, J)"| G_Data[(GINO Data)]
        
        %% Grid(FNO/RNN) Logic
        Scatter --> MeshGrid[mesh_to_grid]
        MeshGrid --> |"Remove A/J from Inputs!\nInputs: 9 -> 7 Channels\nTargets: 2 -> 4 Channels"| Grid_Data[(Grid Data)]
    end

    H5 --> Scatter

    %% 3. Training Scripts
    subgraph Phase 4: Training (Models)
        direction TB
        G_Data --> TG[train_doe_gino.py\n out_channels: 4]
        Grid_Data --> TF[train_doe_fno.py\n out_channels: 4]
        Grid_Data --> TR[train_doe_rnn.py\n out_channels: 4]

        TG -.-> |"Save"| W1((GINO Checkpoint))
        TF -.-> |"Save"| W2((FNO Checkpoint))
        TR -.-> |"Save"| W3((RNN Checkpoint))
    end

    %% 4. Inference Pipeline
    subgraph Phase 3: infer_all_steps_nodes.py
        direction TB
        W1 --> Infer
        W2 --> Infer
        W3 --> Infer
        Infer{Inference Script} -->|Extract pred[:,2] for A\nExtract pred[:,3] for J| NPZ((field_compare_nodes_allsteps.npz))
    end

    %% 5. Visualization GUI
    subgraph Check: paper_visualizations.ipynb
        NPZ --> Jupyter[Load Predictions]
        Jupyter --> GUI[pyMCAD GUI Plot\n A/J Values Successfully Rendered!]
    end