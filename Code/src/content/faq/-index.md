---
title: Frequently Asked Questions
description: "Questions about setup, maps, point linking, and suggestion workflow."
draft: false
faqs:
  - title: Where can I find a tutorial and the full manual?
    answer: Start with the tutorial at [/docs/tutorial](/docs/tutorial) for a guided workflow, then use the manual at [/docs/manual](/docs/manual) for detailed reference of tools and analysis views.

  - title: Which map sources and file formats are supported?
    answer: You can work with uploaded raster maps (for example PNG and JPG) and with the OSM background map. For best suggestion quality, use clear scans with enough contrast and map overlap.

  - title: Do I need to create linked points before using "Suggest Links"?
    answer: No, suggestions can run without existing links. However, adding a few correct seed links first usually improves results because the matcher can enforce better geometric consistency.

  - title: Why do I sometimes get "No suggestions found for the current selection"?
    answer: This usually means the selected area has low overlap, low texture, or strong style differences between maps. Try selecting a smaller region with distinctive features, lowering minimum distance, and increasing the maximum number of suggestions.

  - title: How do I review suggested links before accepting them?
    answer: Use the Suggested Links dialog to accept or dismiss each candidate, or click the ghost markers directly on the maps and review them there. You can also use "Accept All" when the batch looks correct.

  - title: Can I import and export control points and links?
    answer: Yes. The File menu supports importing and exporting old points, new points, and linked points as text files so you can continue work across sessions.
---
