---
title: "Project Title"
date: 2024-01-01
draft: false
---

## Overview

A brief description of your project.

<!-- Embed your Plotly plot snippet here -->
<div id="plotDiv"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
  var data = [{
    x: [1, 2, 3, 4],
    y: [10, 15, 13, 17],
    z: [5, 6, 7, 8],
    mode: 'markers',
    marker: {
      size: 8,
      color: [0, 1, 2, 3],
      colorscale: 'Viridis'
    },
    type: 'scatter3d'
  }];
  var layout = {
    title: 'Interactive 3D Plot',
    margin: {l: 0, r: 0, b: 0, t: 40}
  };
  Plotly.newPlot('plotDiv', data, layout);
</script>
