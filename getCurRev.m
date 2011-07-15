function [ rev ] = getCurRev ()
%GETCURREV Summary of this function goes here
%   Detailed explanation goes here

if exist ('.git', 'dir') ~= 7
  rev = 'Not a git checkout';
  return;
end

fd = fopen (fullfile ('.git', 'HEAD'));
curbranch = textscan (fd, '%s');
path = char (curbranch{1}(2))
rev = readFile (fullfile ('.git', path));
fclose (fd);

end

