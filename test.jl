using JuMP
using COPT

function create_markshare2_model()
    # Create model with COPT solver
    model = Model(COPT.Optimizer)
    
    # Define the coefficient matrix A (7x60)
    # Each row corresponds to one constraint, columns are for x15 to x74
    A = [
        74 49 12 93 56 16 39 77 56 73 1 3 68 61 8 55 18 21 57 98 58 57 46 72 6 16 76 21 78 18 11 58 59 25 32 14 16 3 60 12 7 42 98 34 33 16 97 63 66 28 57 19 74 44 45 49 76 74 9 44;
        20 7 68 69 95 64 76 12 45 43 83 15 90 10 96 98 53 1 2 58 24 90 29 57 19 73 89 31 12 34 67 48 11 22 36 78 75 52 95 57 62 94 10 42 89 11 77 85 30 82 20 52 78 6 57 65 79 83 16 67;
        85 47 67 59 84 59 19 8 50 66 5 51 51 64 64 53 61 45 3 76 17 54 13 89 68 57 4 24 96 81 36 54 3 82 33 88 1 29 4 48 51 14 86 64 73 78 45 65 30 52 6 78 9 19 87 73 10 87 33 1;
        13 71 78 84 56 66 8 68 48 28 33 34 8 99 80 74 2 10 96 41 98 74 39 91 85 95 96 1 80 90 97 36 7 69 9 9 93 94 44 36 71 37 72 38 74 89 37 24 88 77 61 80 2 60 87 80 74 42 2 37;
        35 61 66 78 46 89 61 25 55 16 81 35 96 23 83 39 14 53 23 23 93 38 15 20 19 28 79 51 24 6 3 47 61 60 71 63 26 66 71 63 56 32 39 31 64 89 62 68 59 71 48 76 96 74 61 21 46 18 23 24;
        86 8 44 96 64 65 68 53 19 33 28 42 72 39 5 77 37 89 7 78 10 78 10 96 55 1 64 61 63 90 22 78 92 25 24 65 6 68 66 66 1 67 78 21 47 17 89 77 88 54 10 87 88 80 76 9 83 95 86 24;
        41 64 82 24 48 41 29 93 64 39 92 86 64 45 87 34 39 88 99 63 85 48 83 88 85 5 14 31 12 93 55 1 2 22 93 49 35 25 39 1 77 43 7 42 36 63 5 8 43 18 60 47 47 46 45 38 9 37 8 82
    ]
    
    # RHS values
    b = [1324, 1554, 1429, 1686, 1482, 1613, 1424]
    
    # Define variables
    @variable(model, y[1:7] >= 0)  # Deficit variables (x1, x3, x5, x7, x9, x11, x13)
    @variable(model, x[1:60], Bin)  # Binary variables (x15 to x74 in original)
    
    # Define constraints
    # Each constraint is: sum of A[i,j]*x[j] + y[i] = b[i]
    @constraint(model, con[i=1:7], 
        sum(A[i,j] * x[j] for j in 1:60) + y[i] == b[i]
    )
    
    # Objective: minimize sum of deficits
    @objective(model, Min, sum(y))
    
    return model, x, y
end

# Create and solve the model
model, x, y = create_markshare2_model()

# Optional: Set COPT parameters to match your experiments
# Turn off heuristics and cuts to see the faster performance
# set_optimizer_attribute(model, "Heuristics", 0)
# set_optimizer_attribute(model, "Cuts", 0)

# Solve the model
optimize!(model)

# Print results
println("Solution status: ", termination_status(model))
println("Objective value (total deficit): ", objective_value(model))
println("Nodes explored: ", node_count(model))
println("Solve time: ", solve_time(model))

# Print which binary variables are selected
println("\nSelected binary variables (x15-x74):")
selected = findall(v -> value(v) > 0.5, x)
println("Indices (add 14 for original numbering): ", selected)

# Print deficits for each constraint
println("\nDeficits by constraint:")
for i in 1:7
    println("Constraint $i deficit: ", value(y[i]))
end
