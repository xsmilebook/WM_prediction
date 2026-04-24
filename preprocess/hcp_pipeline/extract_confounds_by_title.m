function extract_confounds_by_title(filePath, savePath_customCP, titles)  
    addpath('/ibmgpfs/cuizaixu_lab/zhaoshaoling/NMF_NeuronCui/Collaborative_Brain_Decomposition-master/lib/NIfTI_20140122');
    % Read the TSV file  
    tbl = readtable(filePath, 'FileType', 'text', 'Delimiter', '\t');  
    % replace 'n/a' with '0'
    for col = 1:width(tbl)
        if iscell(tbl{:, col})
            tbl{strcmp(tbl{:, col}, 'n/a'), col} = {0};
        end
    end
    
    %%
    % Initialize an empty table to store the extracted columns  
    extractedTbl = table;  
    % Iterate through all titles to extract the corresponding columns  
    for i = 1:length(titles)  
        % Find the column with a matching title  
        colIndex = find(strcmp(tbl.Properties.VariableNames, titles{i}));  
        % If a matching column is found, add it to the extracted table  
        if ~isempty(colIndex)  
            extractedTbl.(titles{i}) = tbl{:, colIndex};  
        else  
            warning('Column "%s" not found in the TSV file.', titles{i});  
        end  
    end  

    % Write the extracted table to a new TSV file  
    writetable(extractedTbl, savePath_customCP, 'FileType', 'text', 'Delimiter', '\t');  
    % Display the path of the output file  
    fprintf('Extracted columns have been saved to: %s\n', savePath_customCP);  

end
