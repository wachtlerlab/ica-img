function [ creator ] = genCreatorId ()

git_rev = getCurRev();
mat_ver = regexprep(version, ' \(.+\)', '');

creator = ['MATLAB ' mat_ver ' [' git_rev(1:7) ']'];

end

