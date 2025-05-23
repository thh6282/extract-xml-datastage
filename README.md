# extract-xml-datastage
## Export Folder from DataStage Format XML

1. Open DataStage Designer.
2. Navigate to the job you want to export.
3. Right-click on the job and select `Export`.
4. Choose `DataStage Components` and select `XML` as the export format.
5. Save the exported XML file to your desired location.

## Extract File XML to Mapping

1. Open your XML editor or any tool that can parse XML.
2. Load the exported XML file.
3. Identify the relevant sections that need to be mapped.
4. Create a mapping document that outlines the source XML elements and their corresponding target elements.
5. Save the mapping document for future reference.


## Check syntax query 

## Example Mapping

| Job Name | Stage ID | Stage Name | Stage Type ID | Link Input | Link Output | Source Stage | Target Stage | Query | Query Usage |
|----------|----------|------------|---------------|------------|-------------|--------------|--------------|-------|-------------|
| Job1     | v1s0        | Stage1     | OraclePX         | lnk_a      | lnk_b       | src1      | tgt1      | query1| Select      |
| Job2     | v1s2        | Stage2     | FunnelPX         | lnk_w      | lnk_k       | src2      | tgt2      | query2| AfterSQL      |
| Job3     | v1s3        | Stage3     | Filter         | lnk_new      | lnk_f       | src3      | tgt3      | query3| Select      |


## Notes

- Ensure that all necessary elements are included in the mapping.
- Validate the XML file to ensure it conforms to the required schema.
- Update the mapping document as needed when changes occur.
