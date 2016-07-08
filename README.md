# Minecraft Chunk Editor

This tool can be used to delete chunks from a Minecraft World, remove unused space in region files, create map of visited chunk by date

## Usage
`mche.py --help`

`mche.py [options] <path to world folder>`

## Delete Chunks
You can specify a single chunk, a list of chunk, a zone or a list of zone to delete. The deleted chunks will be regenerated the next time you visit the area.
Coords are given either in chunks coords or blocks coords.

- Syntax for a single coordinate : `3x-1`
The first number is the X coordinate, the second is the Z coordinate.

- Syntax for a list of coordinates : `-1x-7,-3x7`
Chunks coordinates are separated by a comma.

- Syntax for a zone : `0x0_10x10`
The two coordinates are the opposite corner of the rectangle whose chunk's will be deleted.

- Syntax for a list of zones : `0x0_10x10,20x10_20x18`
Zones are separated by a comma.

## Remove unused space
The space used to store a chunk in the region file may vary, when the size decrease, the chunk stored after may not be moved at the end of the current chunk, leaving unused space. Removing those dead space can help you save a few mega bytes (4MB on a 172MB region)

You can see this operation as a kind of defragmentation.

You should probably run this option only when you want to archive your world : the unused space may be used a a later time without having to move all the chunks behind it. When Mojang decided not to save chunks back to back they probably studied the pros and cons and decided it was more efficient to leave space in between to save chunks.

## Gather informations
- Print size occupied by unused space between chunks, you'll see it is pretty marginal most of the time.
- Create gnuplot script and associated data to generate a heatmap of dates at which the chunks were last visisted. Useful to know if you can safely remove a chunk by seeing it was generated years ago. You need gnuplot installed to generate the png.

## Notes : 
- When deleting individual chunks which contains foliage from a nearby chunk not marked for deletion, the regenerated chunk will *not* regenerate the cut-out foliage. This may cause some weird landscape in forests.
- Make backups ! Seriously ! I do not want to be responsible for the loss of your world.
- You can use the --suffix option to specify a file extension used to save modified regions.
- Do not run on a world that is currently opened in Minecraft, unexpected behavior may occurs (chunk out of place)
- If you want to delete a huge zone of chunks that fully contains a region file, the region file will *not* be removed, instead it should result in a 8kB file containing only the region file header with no chunk data.

## FAQ
### Why would I want to delete chunks ?
- If you have a chunk error in your world, get the coords of this chunk and have it generated as it was at the start of your world (you still lose whatever you have constructed)
- Regenerate ores/blocks in your quarry mine without having to go further and further away generating more and more terrain. Beware if you have built nice looking houses on top of your mine.
- Regenerated water temple, villages, end cities

### Why would I want to get informations on the chunks ?
- Because data is beautiful :)

### Why do I have to install gnuplot to get this heatmap ?
- Because I did not want to reinvent the wheel, gnuplot does the job.
- If there is a lot of request, I may consider implementing a native way to generate the heatmap natively.

### Did someone really asked those questions ?
- No, I made them up because I did not know where to put those informations :)
