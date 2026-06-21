import os
import joblib
import networkx as nx
import pandas as pd
from src.utils.logger import logger

class GraphTriangulationDetector:
    def __init__(self, model_path="models/graph_network.pkl"):
        self.model_path = model_path
        self.G = nx.DiGraph()

    def fit(self, df):
        """
        Builds the transactional directed graph from historical transactions.
        df columns must include: [user_id, recipient_id, amount, timestamp]
        If recipient_id is missing, we fall back to generating recipient_id deterministically 
        or using device_id as a proxy node.
        """
        logger.info("Building transactional graph for NetworkX cycle detection...")
        self.G.clear()
        
        # Determine recipient mapping
        if "recipient_id" not in df.columns:
            # Generate deterministic recipient based on device_id to act as the second node
            df = df.copy()
            df["recipient_id"] = df["device_id"].apply(lambda x: f"REC_{x[3:]}" if str(x).startswith("DEV") else f"REC_{x}")

        # Add edges for all historical transactions
        for _, row in df.iterrows():
            sender = row["user_id"]
            receiver = row["recipient_id"]
            self.G.add_edge(sender, receiver, amount=float(row["amount"]), timestamp=str(row["timestamp"]))
            
        # Save graph object
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.G, self.model_path)
        logger.info(f"Transactional graph saved containing {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")
        return self

    def load_model(self):
        """
        Loads graph from disk.
        """
        if os.path.exists(self.model_path):
            self.G = joblib.load(self.model_path)
            logger.info(f"Transactional graph loaded containing {self.G.number_of_nodes()} nodes.")
        else:
            logger.warning(f"Transactional graph not found at {self.model_path}. Fitting is required.")
        return self

    def predict_score(self, tx_dict):
        """
        Determines if the current transaction completes a cycle (triangulation)
        between accounts.
        Returns a risk score of 100 if a triangulation cycle is completed, otherwise 0.
        """
        sender = tx_dict["user_id"]
        
        # Handle recipient mapping fallback
        if "recipient_id" in tx_dict:
            receiver = tx_dict["recipient_id"]
        elif "device_id" in tx_dict:
            dev = tx_dict["device_id"]
            receiver = f"REC_{dev[3:]}" if str(dev).startswith("DEV") else f"REC_{dev}"
        else:
            receiver = "REC_UNKNOWN"

        cycle_detected = False
        cycle_type = "None"
        path = []

        # Check if nodes exist in the graph to check cycles
        if self.G.has_node(sender) and self.G.has_node(receiver):
            # 1. Check for Length-2 cycle: sender -> receiver -> sender
            # (Does receiver have an edge back to sender?)
            if self.G.has_edge(receiver, sender):
                cycle_detected = True
                cycle_type = "Length-2 Cycle (Direct Loop)"
                path = [sender, receiver, sender]
            
            # 2. Check for Length-3 cycle (Triangulation): sender -> receiver -> account_c -> sender
            # (Is there a node C such that receiver -> C and C -> sender?)
            if not cycle_detected:
                receiver_successors = set(self.G.successors(receiver))
                sender_predecessors = set(self.G.predecessors(sender))
                
                # Intersection represents the intermediate nodes C
                intersection = receiver_successors.intersection(sender_predecessors)
                if intersection:
                    cycle_detected = True
                    cycle_type = "Length-3 Cycle (Account Triangulation)"
                    intermediate_node = list(intersection)[0]
                    path = [sender, receiver, intermediate_node, sender]

        # Add this edge to the graph dynamically for future checks (updating streaming state)
        self.G.add_edge(sender, receiver, amount=float(tx_dict["amount"]), timestamp=str(tx_dict["timestamp"]))

        # Risk score calculation
        risk_score = 90.0 if cycle_detected else 0.0
        
        details = {
            "cycle_detected": cycle_detected,
            "cycle_type": cycle_type,
            "cycle_path": path,
            "graph_score": risk_score
        }

        return risk_score, details
