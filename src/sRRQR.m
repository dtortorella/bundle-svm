% sRRQR.m
% Copyright (c) 2018, Xin Xing. All rights reserved.
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% * Redistributions of source code must retain the above copyright notice, this
%   list of conditions and the following disclaimer.
% * Redistributions in binary form must reproduce the above copyright notice,
%   this list of conditions and the following disclaimer in the documentation
%   and/or other materials provided with the distribution
% * Neither the name of Georgia Institute of Technology nor the names of its
%   contributors may be used to endorse or promote products derived from this
%   software without specific prior written permission.
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function [Q,R,p] = sRRQR(A, f, type, par)
   
%   Strong Rank Revealing QR   
%       A P = [Q1, Q2] * [R11, R12; 
%                           0, R22]
%   where R11 and R12 satisfy that (inv(R11) * R12) has entries
%   bounded by a pre-specified constant which is not less than 1. 
%   
%   Input: 
%       A, matrix, target matrix that is appoximated.
%       f, scalar, constant that bounds the entries of calculated (inv(R11) * R12)
%    type, string, be either "rank" or "tol" and specify the way to decide
%                  the dimension of R11, i.e., the truncation. 
%     par, scalar, the parameter for "rank" or "tol" defined by the type. 
%
%   Output: 
%       A(:, p) = [Q1, Q2] * [R11, R12; 
%                               0, R22]
%               approx Q1 * [R11, R12];
%       Only truncated QR decomposition is returned as 
%           Q = Q1, 
%           R = [R11, R12];
%   
%   Reference: 
%       Gu, Ming, and Stanley C. Eisenstat. "Efficient algorithms for 
%       computing a strong rank-revealing QR factorization." SIAM Journal 
%       on Scientific Computing 17.4 (1996): 848-869.
%
%   Note: 
%       1. For a given rank (type = 'rank'), algorithm 4 in the above ref.
%       is implemented.
%       2. For a given error threshold (type = 'tol'), algorithm 6 in the
%       above ref. is implemented. 


%   given a fixed rank 
if (strcmp(type, 'rank'))
    [Q,R,p] = sRRQR_rank(A, f, par);
    return ;
end

%   given a fixed error threshold
if (strcmp(type, 'tol'))
    [Q,R,p] = sRRQR_tol(A, f, par);
    return ;
end

%   report error otherwise
fprintf('No parameter type of %s !\n', type)
Q = [];
R = [];
p = [];

end