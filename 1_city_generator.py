import os
import json
import random
import math
import networkx as nx
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = "/workspace/Maps-Agent-Audit/data/cities"
os.makedirs(DATA_DIR, exist_ok=True)

LANDMARKS = [
    "City Hall", "Central Library", "General Hospital", "National Museum", 
    "Grand Park", "Main Station", "University", "Stadium", "Airport", 
    "Shopping Mall", "Harbor", "Tech Hub", "Financial Center",
    "Police HQ", "Fire Station", "Convention Center", "Zoo", "Aquarium"
]

ZONE_COLORS = {
    "residential": (255, 255, 230), # Light Yellow
    "park": (220, 255, 220),        # Light Green
    "commercial": (220, 240, 255),  # Light Blue
    "industrial": (230, 230, 230),  # Light Grey
    "water": (160, 210, 240),       # River Blue
}

def generate_graph(complexity):
    # Base 10x10
    G = nx.grid_2d_graph(10, 10)
    
    if complexity == 2:
        # Level 2: +3 to 5 diagonals
        nodes = list(G.nodes())
        num_diags = random.randint(3, 5)
        added = 0
        while added < num_diags:
            u = random.choice(nodes)
            v = (u[0] + random.choice([-1, 1]), u[1] + random.choice([-1, 1]))
            if v in nodes and not G.has_edge(u, v):
                G.add_edge(u, v)
                added += 1
                
    elif complexity == 3:
        # Level 3: -25 to 30 edges
        num_remove = random.randint(25, 30)
        edges = list(G.edges())
        random.shuffle(edges)
        removed = 0
        for e in edges:
            if removed >= num_remove:
                break
            # Temporarily remove and check connectivity
            G.remove_edge(*e)
            if not nx.is_connected(G):
                # Put it back if it breaks connectivity
                G.add_edge(*e)
            else:
                removed += 1
                
        # Perturb node coordinates for organic look
        mapping = {}
        for x, y in G.nodes():
            dx = random.uniform(-0.3, 0.3)
            dy = random.uniform(-0.3, 0.3)
            mapping[(x,y)] = (x + dx, y + dy)
        nx.relabel_nodes(G, mapping, copy=False)

    return G

def get_zone_at(x, y, zones_info, water_type, river_y, lake_centers):
    # Check water first
    if water_type == "river":
        river_thickness = 0.5
        river_wave = math.sin(x) * 0.4
        if abs(y - (river_y + river_wave)) < river_thickness:
            return "water"
    elif water_type == "lakes":
        for cx, cy, radius in lake_centers:
            if (x - cx)**2 + (y - cy)**2 < radius**2:
                return "water"
                
    # Otherwise check land zones
    min_dist = float('inf')
    best_zone = "residential"
    for zx, zy, ztype in zones_info:
        dist = (x - zx)**2 + (y - zy)**2
        if dist < min_dist:
            min_dist = dist
            best_zone = ztype
    return best_zone

def validate_graph(G, start_node, end_node):
    if not nx.is_connected(G):
        return False
    try:
        paths = list(nx.node_disjoint_paths(G, start_node, end_node))
        if len(paths) < 2:
            return False
    except nx.NetworkXNoPath:
        return False
    return True

def generate_city(city_id, complexity):
    while True:
        G = generate_graph(complexity)
        
        nodes = list(G.nodes())
        landmark_nodes = random.sample(nodes, 10)
        start_node, end_node = random.sample(landmark_nodes, 2)
        
        if validate_graph(G, start_node, end_node):
            break

    city_landmarks = random.sample(LANDMARKS, 10)
    landmark_mapping = {node: name for node, name in zip(landmark_nodes, city_landmarks)}
    nx.set_node_attributes(G, landmark_mapping, 'landmark')
    
    # Water generation logic
    water_type = random.choice(["river", "lakes", "none"])
    river_y = random.uniform(2, 7) if water_type == "river" else 0
    lake_centers = []
    if water_type == "lakes":
        for _ in range(random.randint(1, 3)):
            lake_centers.append((random.uniform(1, 9), random.uniform(1, 9), random.uniform(0.5, 1.8)))
            
    # Zone Generation logic
    zones_info = []
    for _ in range(6):
        zx, zy = random.uniform(0, 9), random.uniform(0, 9)
        ztype = random.choice(["residential", "commercial", "industrial", "park"])
        zones_info.append((zx, zy, ztype))
    
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0 # PURE PHYSICAL DISTANCE. NO BRIDGE PENALTY.
        
        # Determine if it's visually a bridge for rendering purposes
        mid_x = (u[0] + v[0]) / 2
        mid_y = (u[1] + v[1]) / 2
        if get_zone_at(mid_x, mid_y, zones_info, water_type, river_y, lake_centers) == "water":
            G[u][v]['is_bridge'] = True
        else:
            G[u][v]['is_bridge'] = False

    # Compute ALL shortest paths of equal minimum length for fair Jaccard scoring
    all_shortest = list(nx.all_shortest_paths(G, source=start_node, target=end_node, weight='weight'))
    shortest_path = all_shortest[0]  # used for rendering the ground-truth green line
    
    # Generate sequential labels for nodes.
    # Nodes are sorted by (rounded_row, rounded_col) for deterministic ordering
    # across all complexity levels, including Level 3 irregular graphs.
    # Intersections: i01, i02, i03 ...
    # Dead ends:     d01, d02, d03 ...
    non_landmark_nodes = [n for n in G.nodes() if n not in landmark_mapping]
    non_landmark_nodes.sort(key=lambda n: (int(round(n[0])), int(round(n[1]))))

    intersection_counter = 1
    dead_end_counter = 1
    node_labels = {}
    for node in G.nodes():
        if node in landmark_mapping:
            node_labels[node] = landmark_mapping[node]

    for node in non_landmark_nodes:
        if G.degree(node) == 1:
            node_labels[node] = f"d{dead_end_counter:02d}"
            dead_end_counter += 1
        else:
            node_labels[node] = f"i{intersection_counter:02d}"
            intersection_counter += 1
            
    # Build label lists for every equivalent shortest path
    all_path_names     = [[node_labels[n] for n in p] for p in all_shortest]
    all_path_landmarks = [[landmark_mapping[n] for n in p if n in landmark_mapping] for p in all_shortest]
    # Keep first path as primary for backward-compat fields
    path_names     = all_path_names[0]
    path_landmarks = all_path_landmarks[0]
    num_dead_ends  = sum(1 for n in G.nodes() if G.degree(n) == 1)
    
    centrality = nx.edge_betweenness_centrality(G, weight='weight')
    max_cent = max(centrality.values()) if centrality else 1
    
    # Render
    img_size = 800
    img = Image.new('RGB', (img_size, img_size), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    for py in range(0, img_size, 10):
        for px in range(0, img_size, 10):
            gx = (px - 50) / 77.7
            gy = (py - 50) / 77.7
            ztype = get_zone_at(gx, gy, zones_info, water_type, river_y, lake_centers)
            draw.rectangle([px, py, px+10, py+10], fill=ZONE_COLORS[ztype])
    
    margin = 50
    usable_size = img_size - 2 * margin
    cell_size = usable_size / 9
    
    def node_to_pixel(node):
        x, y = node
        return (margin + x * cell_size, margin + y * cell_size)
    
    # Draw roads
    for u, v in G.edges():
        px_u = node_to_pixel(u)
        px_v = node_to_pixel(v)
        cent = centrality.get((u, v), centrality.get((v, u), 0))
        width = 3 if (cent / max_cent) > 0.2 else 1
        color = (100, 80, 50) if G[u][v]['is_bridge'] else (120, 120, 120)
        draw.line([px_u, px_v], fill=color, width=width)
            
    # Ground truth
    for i in range(len(shortest_path) - 1):
        u = shortest_path[i]
        v = shortest_path[i+1]
        px_u = node_to_pixel(u)
        px_v = node_to_pixel(v)
        draw.line([px_u, px_v], fill=(40, 200, 40), width=2)
    
    # Draw nodes and labels
    for node in G.nodes():
        px = node_to_pixel(node)
        is_endpoint = node in (start_node, end_node)
        is_dead_end = G.degree(node) == 1 and node not in landmark_mapping
        
        if node in landmark_mapping:
            name = landmark_mapping[node]
            if is_endpoint:
                color = (255, 0, 0) if node == start_node else (0, 0, 255)
            elif any(x in name for x in ["Hospital", "Station", "Airport", "Police", "Fire"]):
                color = (220, 40, 40)
            elif any(x in name for x in ["Park", "Zoo", "Aquarium"]):
                color = (40, 200, 40)
            elif any(x in name for x in ["Museum", "Library", "Convention"]):
                color = (150, 40, 200)
            else:
                color = (40, 100, 220)
                
            draw.ellipse([px[0]-8, px[1]-8, px[0]+8, px[1]+8], fill=color)
        elif is_dead_end:
            # Draw red X
            draw.line([px[0]-4, px[1]-4, px[0]+4, px[1]+4], fill=(255,0,0), width=2)
            draw.line([px[0]-4, px[1]+4, px[0]+4, px[1]-4], fill=(255,0,0), width=2)
        else:
            # Normal intersection
            draw.ellipse([px[0]-3, px[1]-3, px[0]+3, px[1]+3], fill=(80, 80, 80))

        # Draw text label with smart boundary-aware positioning
        label = node_labels[node]
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        except Exception:
            font = ImageFont.load_default()

        # Measure text dimensions to avoid overflow
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if node in landmark_mapping:
            # Default: right of dot, above it
            tx = px[0] + 10
            ty = px[1] - text_h - 4
            # Flip LEFT if text would overflow right edge
            if tx + text_w > img_size - 5:
                tx = px[0] - text_w - 10
            # Drop BELOW if text would overflow top edge
            if ty < 2:
                ty = px[1] + 10
        else:
            # Intersection / dead end: always below and to the right
            tx = px[0] + 5
            ty = px[1] + 5
            # Flip LEFT if overflow right edge
            if tx + text_w > img_size - 5:
                tx = px[0] - text_w - 5

        # White outline for contrast, then black text
        for ox, oy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
            draw.text((tx+ox, ty+oy), label, fill=(255, 255, 255), font=font)
        draw.text((tx, ty), label, fill=(0, 0, 0), font=font)

    img_path = os.path.join(DATA_DIR, f"city_{city_id:03d}.png")
    img.save(img_path)
    
    q_size = 425
    overlap = 50
    offset = q_size - overlap
    
    quadrants = {
        'NW': img.crop((0, 0, q_size, q_size)),
        'NE': img.crop((offset, 0, img_size, q_size)),
        'SW': img.crop((0, offset, q_size, img_size)),
        'SE': img.crop((offset, offset, img_size, img_size)),
    }
    
    for name, q_img in quadrants.items():
        q_path = os.path.join(DATA_DIR, f"city_{city_id:03d}_{name}.png")
        q_img.save(q_path)
    
    metadata = {
        "city_id": city_id,
        "complexity_level": complexity,
        "water_type": water_type,
        "num_dead_ends": num_dead_ends,
        "landmarks": {name: list(node) for node, name in landmark_mapping.items()},
        "start": landmark_mapping[start_node],
        "end": landmark_mapping[end_node],
        "shortest_path_nodes": [list(n) for n in shortest_path],
        "shortest_path_names": path_names,
        "shortest_path_landmarks": path_landmarks,
        "all_shortest_paths_names": all_path_names,
        "all_shortest_paths_landmarks": all_path_landmarks,
        "num_equivalent_paths": len(all_shortest),
        "path_length": len(shortest_path) - 1
    }
    
    with open(os.path.join(DATA_DIR, f"city_{city_id:03d}_meta.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    return metadata

if __name__ == "__main__":
    print("Generating 120 verified cities across 3 complexity levels (40 each)...")
    for i in range(1, 121):
        if i <= 40:
            comp = 1
        elif i <= 80:
            comp = 2
        else:
            comp = 3
        generate_city(i, comp)
        if i % 10 == 0:
            print(f"Generated {i}/120 cities.")
    print("Done!")
