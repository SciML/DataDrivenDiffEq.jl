
function _gradient(f)
    function j(x)
        ForwardDiff.gradient(f, x)
    end
end
