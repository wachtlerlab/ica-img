function [ data ] = readFile (path)
%READFILE Read in contents of a file

fd = fopen(path);
data = textscan (fd,'%s','Delimiter','\n');
data = char (data{1});
fclose (fd);

end

