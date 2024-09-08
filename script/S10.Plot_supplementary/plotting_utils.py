


# Generate 11 distinct colors with increased lightness and saturation
def generate_diverging_colors():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8']
    colors = []
    for color in base_colors:
        rgb_color = mcolors.hex2color(color)
        hsv_color = mcolors.rgb_to_hsv(rgb_color)
        h, s, v = hsv_color
        s = min(1, s * 1.1)  # Increase saturation
        v = min(1, v * 1)  # Increase value (lightness)
        colors.append(mcolors.hsv_to_rgb([h, s, v]))
    return colors

# # Generate diverging colors
# diverging_colors = generate_diverging_colors()

# # Print the hexadecimal color codes
# for i, color in enumerate(diverging_colors):
#     hex_code = mcolors.to_hex(color)
#     print(f"Color {i+1}: {hex_code}")

# # Plot the colors for visualization
# plt.figure(figsize=(8, 2))
# for i, color in enumerate(diverging_colors):
#     plt.fill_between([i, i+1], 0, 1, color=color)
# plt.axis('off')
# plt.show()



def get_color_dict():
    diverging_colors = generate_diverging_colors()
    
    color_dict = {
        'all': diverging_colors[0], 
        'Trophic_Niche_Aquatic_predator': diverging_colors[9], 
        'Trophic_Niche_Frugivore': diverging_colors[8], 
        'Trophic_Niche_Granivore': diverging_colors[1], 
        'Trophic_Niche_Herbivore_aquatic': diverging_colors[10], 
        'Trophic_Niche_Herbivore_terrestrial': diverging_colors[4], 
        'Trophic_Niche_Invertivore': diverging_colors[2], 
        'Trophic_Niche_Nectarivore': diverging_colors[6], 
        'Trophic_Niche_Omnivore': diverging_colors[7], 
        'Trophic_Niche_Scavenger':diverging_colors[3], 
        'Trophic_Niche_Vertivore':diverging_colors[5]
    }
    
    return color_dict

def name_processor(niche):

    if niche=='all':
        name='All Species'
    else:
        name=niche.split('Trophic_Niche_')[-1]
        name = name.replace('_',' ')
        if 'Herbivore' in name:
            b = name.split(' ')[-1]
            a = name.split(' ')[0]
            name = b.capitalize()+' '+a.lower()
    return name













