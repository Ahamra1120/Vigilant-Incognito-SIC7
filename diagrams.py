# diagrams.py - File terpisah untuk diagram
import streamlit as st
import graphviz as gv

class VigilantDiagrams:
    """Class untuk menghasilkan diagram Vigilant"""
    
    @staticmethod
    def get_flowchart():
        return create_flowchart()
    
    @staticmethod
    def get_schematic():
        return create_circuit_schematic()
    
    @staticmethod
    def get_system_architecture():
        """Membuat diagram arsitektur sistem"""
        dot = gv.Digraph(comment='System Architecture')
        dot.attr(rankdir='LR', bgcolor='#0f172a', fontcolor='white')
        
        # Nodes
        dot.node('sensors', '📡 Sensors\nDHT11, Camera', 
                shape='box', fillcolor='#3b82f6')
        dot.node('esp32', '⚡ ESP32\nMicrocontroller', 
                shape='ellipse', fillcolor='#f59e0b')
        dot.node('wifi', '📶 WiFi\nConnection', 
                shape='box', fillcolor='#0ea5e9')
        dot.node('mqtt', '📨 MQTT\nBroker', 
                shape='box', fillcolor='#8b5cf6')
        dot.node('server', '🖥️ Server\nStreamlit + ML', 
                shape='cylinder', fillcolor='#10b981')
        dot.node('ui', '🎨 Dashboard\nStreamlit UI', 
                shape='component', fillcolor='#ef4444')
        dot.node('storage', '💾 Storage\nDatabase & Logs', 
                shape='folder', fillcolor='#64748b')
        
        # Connections
        dot.edges([
            ('sensors', 'esp32'),
            ('esp32', 'wifi'),
            ('wifi', 'mqtt'),
            ('mqtt', 'server'),
            ('server', 'ui'),
            ('server', 'storage'),
            ('ui', 'storage')
        ])
        
        return dot

# Fungsi untuk export diagram ke file
def export_diagrams():
    """Export diagrams to files"""
    diagrams = VigilantDiagrams()
    
    # Export flowchart
    flowchart = diagrams.get_flowchart()
    flowchart.render('assets/flowchart_detailed', format='png', cleanup=True)
    
    # Export architecture
    architecture = diagrams.get_system_architecture()
    architecture.render('assets/system_architecture', format='png', cleanup=True)