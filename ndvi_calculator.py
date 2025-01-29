import streamlit as st
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
import plotly.express as px

def calculate_ndvi(red_band, nir_band):
    return (nir_band - red_band) / (nir_band + red_band + 1e-10)

def process_bands(red_band_path, nir_band_path, satellite_choice):
    with rasterio.open(red_band_path) as red_src, rasterio.open(nir_band_path) as nir_src:
        red_band = red_src.read(1)
        nir_band = nir_src.read(1)
        
        red_nodata = red_src.nodata
        nir_nodata = nir_src.nodata
        
        red_band = np.ma.masked_equal(red_band, red_nodata)
        nir_band = np.ma.masked_equal(nir_band, nir_nodata)
        
        if satellite_choice == "Sentinel-2":
            red_band = red_band / 10000.0
            nir_band = nir_band / 10000.0
        elif satellite_choice == "Landsat 8/9":
            red_band = red_band * 0.0000275 - 0.2
            nir_band = nir_band * 0.0000275 - 0.2
        
        ndvi = calculate_ndvi(red_band, nir_band)
        ndvi = np.clip(ndvi, -1, 1)
        
        return ndvi, red_src

def main():
    st.image("banner_ndvi.jpg", use_container_width=True)
    st.title("NDVI Calculator")
    st.markdown("Analyze vegetation cover with satellite data.")
    st.markdown("---")
    
    if "ndvi" not in st.session_state:
        st.session_state.ndvi = None
    
    satellite_choice = st.selectbox("Select Satellite Type", ["Landsat 8/9", "Sentinel-2"])
    
    st.subheader("Upload Satellite Bands")
    if satellite_choice == "Landsat 8/9":
        red_band = st.file_uploader("Upload Band 4 (Red)", type=["tif"])
        nir_band = st.file_uploader("Upload Band 5 (NIR)", type=["tif"])
    else:
        red_band = st.file_uploader("Upload Band 4 (Red)", type=["tif", "jp2"])
        nir_band = st.file_uploader("Upload Band 8 (NIR)", type=["tif", "jp2"])
    
    classifications = [
        (-1.0, -0.1, "Water"),
        (-0.1, 0.1, "Barren (Rock/Sand/Snow)"),
        (0.1, 0.4, "Shrub & Grassland"),
        (0.4, 1.0, "Temperate/Tropical Rainforest"),
    ]
    
    if st.button("Calculate NDVI"):
        if red_band and nir_band:
            try:
                st.session_state.ndvi, red_src = process_bands(red_band, nir_band, satellite_choice)
                st.success("NDVI calculation complete!")
            except Exception as error:
                st.error(f"Error: {error}")
        else:
            st.warning("Please upload all required bands to calculate NDVI.")
    
    if st.session_state.ndvi is not None and st.button("Display NDVI Map"):
        ndvi_scaled = (st.session_state.ndvi + 1) * 127.5
        ndvi_scaled = ndvi_scaled.astype(np.uint8)
        
        ndvi_resized = Image.fromarray(ndvi_scaled).resize((512, 512), Image.NEAREST)
        ndvi_resized = np.array(ndvi_resized)
        
        ndvi_resized = (ndvi_resized / 127.5) - 1
        
        classification_labels = np.zeros_like(ndvi_resized, dtype=object)
        for lower, upper, label in classifications:
            mask = (ndvi_resized >= lower) & (ndvi_resized < upper)
            classification_labels[mask] = label
        
        fig = px.imshow(
            ndvi_resized,
            color_continuous_scale="RdYlGn",
            labels={"color": "NDVI Value"},
            title="Interactive NDVI Map",
            range_color=[-1, 1],
        )
        fig.update_traces(
            hovertemplate="<b>NDVI Value:</b> %{z:.2f}<br><b>Classification:</b> %{customdata}",
            customdata=classification_labels,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.ndvi is not None and st.button("Produce NDVI Histogram"):
        ndvi_sampled = st.session_state.ndvi.flatten()
        ndvi_sampled = ndvi_sampled[~np.isnan(ndvi_sampled)]
        
        bins = [lower for lower, _, _ in classifications] + [classifications[-1][1]]
        labels = [label for _, _, label in classifications]
        ndvi_categories = pd.cut(ndvi_sampled, bins=bins, labels=labels, include_lowest=True, right=False)
        
        category_counts = ndvi_categories.value_counts().reindex(labels, fill_value=0)
        
        hist_fig = px.bar(
            category_counts,
            x=category_counts.index.astype(str),
            y=category_counts.values,
            labels={"x": "NDVI Classification", "y": "Frequency"},
            title="NDVI Classification Distribution",
        )
        st.plotly_chart(hist_fig, use_container_width=True)
    
    if st.session_state.ndvi is not None and st.button("Download NDVI as CSV"):
        ndvi_flat = st.session_state.ndvi.flatten()
        ndvi_flat = ndvi_flat[~np.isnan(ndvi_flat)]
        
        sampled_indices = np.random.choice(len(ndvi_flat), size=min(100000, len(ndvi_flat)), replace=False)
        sampled_ndvi = ndvi_flat[sampled_indices]
        ndvi_df = pd.DataFrame(sampled_ndvi, columns=["NDVI"])
        csv = ndvi_df.to_csv(index=False)
        st.download_button(
            label="Download NDVI as CSV",
            data=csv,
            file_name="ndvi_values.csv",
            mime="text/csv",
        )
    
    st.markdown("---")
    st.markdown("Powered by Streamlit | NDVI Analysis Tool")

if __name__ == "__main__":
    main()