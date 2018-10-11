@testset "gp basics" begin

    # Test the creation of indepenent GPs
    let

        # Create two independent GPs
        mu1, mu2 = ConstantMean(1.0), ZeroMean{Float64}()
        k1, k2 = EQ(), PerEQ(0.5)
        f1, f2 = GP(mu1, k1), GP(mu2, k2)

        # @test mean(GP(EQ(), GPC())) == ZeroMean{Float64}()

        # @test mean(f1) == μ1
        # @test mean(f2) == μ2

        # @test kernel(f1) == k1
        # @test kernel(f2) == k2

        # @test kernel(f1, f1) == k1
        # @test kernel(f1, f2) == ZeroKernel{Float64}()
        # @test kernel(f2, f1) == ZeroKernel{Float64}()
        # @test kernel(f2, f2) == k2
    end



end
