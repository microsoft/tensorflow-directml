//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************
// clang-format off

#pragma once
#include "DirectML.h"

#include <cstdint>
#include <cassert>
#include <vector>
#include <array>
#include <deque>
#include <memory>
#include <utility>
#include <optional>
#include <type_traits>

#ifdef DMLX_USE_GSL
#include "gsl/gsl_assert"
#include "gsl/span"
#endif

/** Calculates the minimum number of bytes required to store a buffer tensor with the specified type, sizes, and
    strides. The formula can be expressed as the following:

    IndexOfLastElement = dot(Sizes - 1, Strides);
    MinimumImpliedSizeInBytes = roundup((IndexOfLastElement + 1) * ElementSizeInBytes, 4)

    In other words, the minimum size of a tensor is the index of the one-past-the-end element, multiplied by the
    element size (e.g. 2 bytes for a FLOAT16 tensor). Additionally DirectML requires that all buffers bound must have
    a total size which is DWORD-aligned, and hence the minimum implied size in bytes must be rounded up to the nearest
    4-byte boundary.
    */

inline UINT64 DMLCalcBufferTensorSize(
    DML_TENSOR_DATA_TYPE dataType,
    UINT dimensionCount,
    _In_reads_(dimensionCount) const UINT* sizes,
    _In_reads_opt_(dimensionCount) const UINT* strides)
{
    UINT elementSizeInBytes = 0;
    switch (dataType)
    {
    case DML_TENSOR_DATA_TYPE_FLOAT32:
    case DML_TENSOR_DATA_TYPE_UINT32:
    case DML_TENSOR_DATA_TYPE_INT32:
        elementSizeInBytes = 4;
        break;

    case DML_TENSOR_DATA_TYPE_FLOAT16:
    case DML_TENSOR_DATA_TYPE_UINT16:
    case DML_TENSOR_DATA_TYPE_INT16:
        elementSizeInBytes = 2;
        break;

    case DML_TENSOR_DATA_TYPE_UINT8:
    case DML_TENSOR_DATA_TYPE_INT8:
        elementSizeInBytes = 1;
        break;

    case DML_TENSOR_DATA_TYPE_FLOAT64:
    case DML_TENSOR_DATA_TYPE_UINT64:
    case DML_TENSOR_DATA_TYPE_INT64:
        elementSizeInBytes = 8;
        break;

    default:
        return 0; // Invalid data type
    }

    UINT64 minimumImpliedSizeInBytes = 0;
    if (!strides)
    {
        minimumImpliedSizeInBytes = sizes[0];
        for (UINT i = 1; i < dimensionCount; ++i)
        {
            minimumImpliedSizeInBytes *= sizes[i];
        }
        minimumImpliedSizeInBytes *= elementSizeInBytes;
    }
    else
    {
        UINT indexOfLastElement = 0;
        for (UINT i = 0; i < dimensionCount; ++i)
        {
            indexOfLastElement += (sizes[i] - 1) * strides[i];
        }

        minimumImpliedSizeInBytes = (indexOfLastElement + 1) * elementSizeInBytes;
    }

    // Round up to the nearest 4 bytes.
    minimumImpliedSizeInBytes = (minimumImpliedSizeInBytes + 3) & ~3ull;

    return minimumImpliedSizeInBytes;
}

namespace dml
{
    namespace detail
    {
        // Provide non-member size() and data(). Defaults to standard library implementation (if available)
    #if __cpp_lib_nonmember_container_access
        template <typename C>
        constexpr auto size(const C& c) -> decltype(c.size())
        {
            return std::size(c);
        }

        template <typename T, std::size_t N>
        constexpr std::size_t size(const T(&array)[N]) noexcept
        {
            return std::size(array);
        }

        template <typename C>
        constexpr auto data(C& c) -> decltype(c.data())
        {
            return std::data(c);
        }

        template <typename T, std::size_t N>
        constexpr T* data(T(&array)[N]) noexcept
        {
            return std::data(array);
        }
    #else
        template <typename C>
        constexpr auto size(const C& c) -> decltype(c.size())
        {
            return c.size();
        }

        template <typename T, std::size_t N>
        constexpr std::size_t size(const T(&array)[N]) noexcept
        {
            return N;
        }

        template <typename C>
        constexpr auto data(C& c) -> decltype(c.data())
        {
            return c.data();
        }

        template <typename T, std::size_t N>
        constexpr T* data(T(&array)[N]) noexcept
        {
            return array;
        }
    #endif

        template <typename T>
        class span
        {
        public:
            span() = default;

            constexpr span(std::initializer_list<T> i) : m_begin(i.begin()), m_end(i.end()) {}
            constexpr span(T* begin, T* end) : m_begin(begin), m_end(end) {}
            constexpr span(T* begin, size_t elementCount) : m_begin(begin), m_end(begin + elementCount) {}

            template <typename ContiguousContainer>
            constexpr span(ContiguousContainer&& container)
              : m_begin(dml::detail::data(container)), m_end(m_begin + dml::detail::size(container)) {}

            template <size_t N>
            constexpr span(T(&a)[N]) noexcept : span(a, N) {}

            T* data() noexcept { return m_begin; }
            T* begin() noexcept { return m_begin; }
            T* end() noexcept { return m_end; }
            T const* data() const noexcept { return m_begin; }
            T const* begin() const noexcept { return m_begin; }
            T const* end() const noexcept { return m_end; }
            bool empty() const noexcept { return m_end == m_begin; }
            size_t size() const noexcept { return m_end - m_begin; }
            size_t size_bytes() const noexcept { return sizeof(T) * size(); }
            T& operator[](size_t index) const noexcept { return m_begin[index]; }
            span<T> subspan(size_t index, size_t count) { return span<T>(m_begin + index, m_begin + index + count); }

        protected:
            T* m_begin = nullptr;
            T* m_end = nullptr;
        };
    }

#if DMLX_USE_ABSEIL 
    template <typename T>
    using Optional = absl::optional<T>;

    constexpr absl::nullopt_t NullOpt = absl::nullopt;

    template <typename T, size_t N>
    using SmallVector = absl::InlinedVector<T, N>;

    template <typename T>
    using Span = absl::Span<T>;

    using absl::make_unique;
#else
    template <typename T>
    using Optional = std::optional<T>;
    
    constexpr std::nullopt_t NullOpt = std::nullopt;

    template <typename T, size_t N>
    using SmallVector = std::vector<T>;

   #ifdef __cpp_lib_span
        template <typename T>
        using Span = std::span<T>;
    #elif DMLX_USE_GSL
        template <typename T>
        using Span = gsl::span<T>;
    #else 
        template <typename T>
        using Span = dml::detail::span<T>;
   #endif

   using std::make_unique;
#endif

#if __cpp_exceptions
    #if DMLX_USE_WIL
        #define DMLX_THROW_IF_FAILED(_hr) THROW_IF_FAILED(_hr)
        #define DMLX_THROW(_hr) THROW_HR(_hr)
    #else
        #define DMLX_THROW_IF_FAILED(_hr) if (FAILED(_hr)) { throw std::runtime_error(#_hr); }
        #define DMLX_THROW(_hr) throw std::runtime_error(#_hr); 
    #endif
#else
    #define DMLX_THROW_IF_FAILED(_hr) if (FAILED(_hr)) { std::abort(); }
    #define DMLX_THROW(_hr) { std::abort(); } 
#endif

    class Scope;
    class Expression;

    enum class TensorLayout
    {
        Default = 0,
        Nchw = 0,
        Nhwc = 1,
    };

    // Helper for computing DML tensor strides for a given layout
    inline void CalculateStrides(Span<const uint32_t> sizes, TensorLayout layout, _Out_ Span<uint32_t> strides)
    {
        uint32_t dimensionCount = static_cast<uint32_t>(sizes.size());
        assert(strides.size() == dimensionCount);

        if (dimensionCount == 4)
        {
            enum DML_ORDER { N, C, H, W };

            switch (layout)
            {
            case TensorLayout::Nchw:
                strides[N] = sizes[C] * sizes[H] * sizes[W];
                strides[C] = sizes[H] * sizes[W];
                strides[H] = sizes[W];
                strides[W] = 1;
                break;

            case TensorLayout::Nhwc:
                strides[N] = sizes[H] * sizes[W] * sizes[C];
                strides[H] = sizes[W] * sizes[C];
                strides[W] = sizes[C];
                strides[C] = 1;
                break;

            default:
                assert(false); // Unrecognized tensor layout
                DMLX_THROW(E_UNEXPECTED);
            }
        }
        else
        {
            assert(dimensionCount == 5);
            
            enum DML_ORDER { N, C, D, H, W };

            switch (layout)
            {
            case TensorLayout::Nchw:
                strides[N] = sizes[C] * sizes[D] * sizes[H] * sizes[W];
                strides[C] = sizes[D] * sizes[H] * sizes[W];
                strides[D] = sizes[H] * sizes[W];
                strides[H] = sizes[W];
                strides[W] = 1;
                break;

            case TensorLayout::Nhwc:
                strides[N] = sizes[D] * sizes[H] * sizes[W] * sizes[C];
                strides[D] = sizes[H] * sizes[W] * sizes[C];
                strides[H] = sizes[W] * sizes[C];
                strides[W] = sizes[C];
                strides[C] = 1;
                break;

            default:
                assert(false); // Unrecognized tensor layout
                DMLX_THROW(E_UNEXPECTED);
            }
        }
    }

    struct TensorDesc
    {
    public:
        using Dimensions = SmallVector<uint32_t, DML_TENSOR_DIMENSION_COUNT_MAX>;

        DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_UNKNOWN;
        DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE;
        Dimensions sizes;
        Optional<Dimensions> strides;
        uint64_t totalTensorSizeInBytes = 0;
        uint32_t guaranteedBaseOffsetAlignment = 0;

        TensorDesc() = default;

        TensorDesc(DML_TENSOR_DATA_TYPE dataType, Dimensions sizes, TensorLayout layout = TensorLayout::Default)
        {
            auto initialStrides = GetStrides(sizes, layout);
            Initialize(dataType, DML_TENSOR_FLAG_NONE, std::move(sizes), std::move(initialStrides));
        }

        TensorDesc(DML_TENSOR_DATA_TYPE dataType, DML_TENSOR_FLAGS flags, Dimensions sizes, TensorLayout layout = TensorLayout::Default)
        {
            auto initialStrides = GetStrides(sizes, layout);
            Initialize(dataType, flags, std::move(sizes), std::move(initialStrides));
        }

        TensorDesc(DML_TENSOR_DATA_TYPE dataType, Dimensions sizes, Optional<Dimensions> strides)
        {
            Initialize(dataType, DML_TENSOR_FLAG_NONE, std::move(sizes), std::move(strides));
        }

        TensorDesc(DML_TENSOR_DATA_TYPE dataType, DML_TENSOR_FLAGS flags, Dimensions sizes, Optional<Dimensions> strides)
        {
            Initialize(dataType, flags, std::move(sizes), std::move(strides));
        }

        /* implicit */ TensorDesc(const DML_TENSOR_DESC& desc)
            : TensorDesc(*static_cast<const DML_BUFFER_TENSOR_DESC*>(desc.Desc))
        {
            assert(desc.Type == DML_TENSOR_TYPE_BUFFER);
            assert(desc.Desc != nullptr);
        }

        /* implicit */ TensorDesc(const DML_BUFFER_TENSOR_DESC& desc)
        {
            this->dataType = desc.DataType;
            this->flags = desc.Flags;
            this->sizes.assign(desc.Sizes, desc.Sizes + desc.DimensionCount);
            if (desc.Strides)
            {
                this->strides.emplace();
                this->strides->assign(desc.Strides, desc.Strides + desc.DimensionCount);
            }
            this->totalTensorSizeInBytes = desc.TotalTensorSizeInBytes;
            this->guaranteedBaseOffsetAlignment = desc.GuaranteedBaseOffsetAlignment;
        }

        // Returns an equivalent DML_TENSOR_DESC or DML_BUFFER_TENSOR_DESC. The returned object contains pointers
        // into the TensorDesc, so it is only valid as long as the TensorDesc itself is alive.
        template <typename T>
        T* AsPtr()
        {
            // "sizeof(T) == -1" is always false; this is just to make the static_assert dependent on the template
            // parameter and therefore not evaluated until template instantiation
            static_assert(sizeof(T) == -1, "Invalid type");
        }
        
        template <>
        DML_BUFFER_TENSOR_DESC* AsPtr<DML_BUFFER_TENSOR_DESC>()
        {
            assert(!strides || sizes.size() == strides->size());

            m_bufferDesc.DataType = this->dataType;
            m_bufferDesc.Flags = this->flags;
            m_bufferDesc.DimensionCount = static_cast<UINT>(sizes.size());
            m_bufferDesc.Sizes = this->sizes.data();
            m_bufferDesc.Strides = this->strides ? this->strides->data() : nullptr;
            m_bufferDesc.TotalTensorSizeInBytes = this->totalTensorSizeInBytes;
            m_bufferDesc.GuaranteedBaseOffsetAlignment = this->guaranteedBaseOffsetAlignment;
            return &m_bufferDesc;
        }
        
        template <>
        DML_TENSOR_DESC* AsPtr<DML_TENSOR_DESC>()
        {
            m_tensorDesc = DML_TENSOR_DESC{ DML_TENSOR_TYPE_BUFFER, AsPtr<DML_BUFFER_TENSOR_DESC>() };
            return &m_tensorDesc;
        }
        
    private:
        DML_BUFFER_TENSOR_DESC m_bufferDesc;
        DML_TENSOR_DESC m_tensorDesc;

        static Optional<Dimensions> GetStrides(const Dimensions& sizes, TensorLayout layout)
        {
            if (layout == TensorLayout::Nchw)
            {
                // NCHW is the default; no need to compute strides
                return NullOpt;
            }

            size_t dimensionCount = sizes.size();
            Dimensions strides(dimensionCount);
            CalculateStrides(sizes, layout, /* out */ Span<uint32_t>(strides.data(), dimensionCount));

            return strides;
        }

        void Initialize(
            DML_TENSOR_DATA_TYPE tensorDataType,
            DML_TENSOR_FLAGS tensorFlags,
            Dimensions tensorSizes,
            Optional<Dimensions> tensorStrides)
        {
            const uint32_t dimensionCount = static_cast<uint32_t>(tensorSizes.size());
            assert(!tensorStrides || tensorStrides->size() == dimensionCount);

            this->dataType = tensorDataType;
            this->flags = tensorFlags;
            this->sizes = std::move(tensorSizes);
            this->strides = std::move(tensorStrides);
            this->totalTensorSizeInBytes = DMLCalcBufferTensorSize(
                this->dataType,
                dimensionCount,
                this->sizes.data(),
                this->strides ? this->strides->data() : nullptr);
            this->guaranteedBaseOffsetAlignment = 0;
        }
    };

    namespace detail
    {
        class GraphBuilder;
        class NodeOutput;

        // A node in the graph which represents a graph input.
        struct InputNode
        {
            uint32_t inputIndex;
        };

        // A node in the graph which represents a DML operator.
        struct OperatorNode
        {
            Microsoft::WRL::ComPtr<IDMLOperator> op;

            // The inputs to this node
            std::vector<NodeOutput*> inputs;
        };

        // Used for representing reshapes and type punning
        struct ReinterpretNode
        {
            NodeOutput* input;
        };

        enum class NodeType
        {
            Invalid,
            Input,
            Operator,
            Reinterpret,
        };

        // Identifies a node in the graph.
        struct NodeID
        {
            NodeType type;
            uint32_t index; // The index of this node in the GraphBuilder
        };

        // Represents one of the outputs of a node.
        class NodeOutput
        {
        public:
            NodeOutput(GraphBuilder* owner, NodeID node, uint32_t outputIndex, TensorDesc tensorDesc)
                : m_owner(owner)
                , m_node(node)
                , m_outputIndex(outputIndex)
                , m_tensorDesc(std::move(tensorDesc))
            {}

            // Retrieves the GraphBuilder that owns this object.
            GraphBuilder* GetGraphBuilder() const { return m_owner; }

            NodeID GetNode() const { return m_node; }
            uint32_t GetOutputIndex() const { return m_outputIndex; }
            const TensorDesc& GetOutputDesc() const { return m_tensorDesc; }

        private:
            GraphBuilder* m_owner;
            NodeID m_node;

            // An operator can have multiple outputs; this index identifies which one of the operator's  outputs this
            // NodeOutput represents.
            uint32_t m_outputIndex;

            TensorDesc m_tensorDesc;
        };

        struct GraphDesc
        {
            uint32_t inputCount;
            uint32_t outputCount;
            std::vector<DML_OPERATOR_GRAPH_NODE_DESC> nodes;
            std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
            std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
            std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        };

        class GraphBuilder
        {
        public:
            GraphBuilder(IDMLDevice* device, TensorLayout outputLayout)
                : m_device(device)
                , m_outputLayout(outputLayout)
            {}

            IDMLDevice* GetDevice() const
            {
                return m_device.Get();
            }

            TensorLayout GetOutputLayout() const
            {
                return m_outputLayout;
            }

            // Creates a DML operator node owned by this graph builder and returns a NodeInfo identifier. The
            // inputs to this node must be supplied in the correct order matching the DML operator.
            NodeID CreateOperatorNode(DML_OPERATOR_TYPE type, const void* desc, Span<NodeOutput* const> inputs);
            NodeID CreateInputNode(uint32_t inputIndex);
            NodeID CreateReinterpretNode(NodeOutput* input);
            NodeOutput* CreateNodeOutput(NodeID node, uint32_t outputIndex, TensorDesc tensorDesc);
            GraphDesc GetGraphDesc(Span<const Expression> outputs) const;

        private:
            Microsoft::WRL::ComPtr<IDMLDevice> m_device;
            TensorLayout m_outputLayout;
            std::vector<InputNode> m_inputNodes;
            std::vector<OperatorNode> m_operatorNodes;
            std::vector<ReinterpretNode> m_reinterpretNodes;
            std::deque<NodeOutput> m_nodeOutputs; // deque doesn't invalidate references to elements when it resizes
        };

    } // namespace detail

    class Scope
    {
    public:
        explicit Scope(IDMLDevice* device, TensorLayout outputLayout = TensorLayout::Default)
            : m_graphBuilder(make_unique<detail::GraphBuilder>(device, outputLayout))
        {}

        // For internal use only
        detail::GraphBuilder* Impl() { return m_graphBuilder.get(); }

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> Compile(
            DML_EXECUTION_FLAGS flags,
            Span<const Expression> outputs) const
        {
            detail::GraphDesc graph = m_graphBuilder->GetGraphDesc(outputs);

            // If there's only a single node, don't bother creating a graph - just compile the operator directly.
            if (graph.nodes.size() == 1)
            {
                IDMLDevice* device = m_graphBuilder->GetDevice();

                Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiledOp;
                DMLX_THROW_IF_FAILED(device->CompileOperator(graph.nodes[0].Operator, flags, IID_PPV_ARGS(&compiledOp)));

                return compiledOp;
            }

            std::vector<DML_GRAPH_NODE_DESC> graphNodes(graph.nodes.size());
            for (size_t i = 0; i < graphNodes.size(); ++i)
            {
                graphNodes[i] = { DML_GRAPH_NODE_TYPE_OPERATOR, &graph.nodes[i] };
            }

            std::vector<DML_GRAPH_EDGE_DESC> inputEdges(graph.inputEdges.size());
            for (size_t i = 0; i < inputEdges.size(); ++i)
            {
                inputEdges[i] = { DML_GRAPH_EDGE_TYPE_INPUT, &graph.inputEdges[i] };
            }

            std::vector<DML_GRAPH_EDGE_DESC> outputEdges(graph.outputEdges.size());
            for (size_t i = 0; i < outputEdges.size(); ++i)
            {
                outputEdges[i] = { DML_GRAPH_EDGE_TYPE_OUTPUT, &graph.outputEdges[i] };
            }

            std::vector<DML_GRAPH_EDGE_DESC> intermediateEdges(graph.intermediateEdges.size());
            for (size_t i = 0; i < intermediateEdges.size(); ++i)
            {
                intermediateEdges[i] = { DML_GRAPH_EDGE_TYPE_INTERMEDIATE, &graph.intermediateEdges[i] };
            }

            DML_GRAPH_DESC graphDesc = {};
            graphDesc.InputCount = graph.inputCount;
            graphDesc.OutputCount = graph.outputCount;
            graphDesc.NodeCount = static_cast<UINT>(graphNodes.size());
            graphDesc.Nodes = graphNodes.data();
            graphDesc.InputEdgeCount = static_cast<UINT>(inputEdges.size());
            graphDesc.InputEdges = inputEdges.data();
            graphDesc.OutputEdgeCount = static_cast<UINT>(outputEdges.size());
            graphDesc.OutputEdges = outputEdges.data();
            graphDesc.IntermediateEdgeCount = static_cast<UINT>(intermediateEdges.size());
            graphDesc.IntermediateEdges = intermediateEdges.data();

            Microsoft::WRL::ComPtr<IDMLDevice1> device1;
            DMLX_THROW_IF_FAILED(m_graphBuilder->GetDevice()->QueryInterface(IID_PPV_ARGS(&device1)));

            Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiledGraph;
            DMLX_THROW_IF_FAILED(device1->CompileGraph(&graphDesc, flags, IID_PPV_ARGS(&compiledGraph)));

            return compiledGraph;
        }

    private:
        std::unique_ptr<detail::GraphBuilder> m_graphBuilder;
    };

    class Expression
    {
    public:
        /*implicit*/ Expression(detail::NodeOutput* nodeOutput = nullptr)
            : m_nodeOutput(nodeOutput)
        {}

        // Returns a struct containing the required properties of the tensor to hold the output of this expression,
        // once evaluated.
        const TensorDesc& GetOutputDesc() const { return Impl()->GetOutputDesc(); }

        // For internal use only
        detail::NodeOutput* Impl() const { return m_nodeOutput; }

    private:
        detail::NodeOutput* m_nodeOutput; // weak; this is owned by the GraphBuilder
    };

    // Represents an activation to be fused with an existing operator. The meaning of param1 and param2 depend on the
    // activation to be fused.
    // 
    // For HARD_SIGMOID, LINEAR, PARAMETRIC_SOFTPLUS, and SCALED_TANH: param1 = Alpha and param2 = Beta
    // For ELU, LEAKY_RELU, and THRESHOLDED_RELU: param1 = Alpha. param2 is unused.
    // For SCALED_ELU, param1 = Alpha and param2 = Gamma.
    // For ACTIVATION_SOFTPLUS, param1 = Steepness.
    // For all other activations, both param1 and param2 are unused.
    struct FusedActivation
    {
        DML_OPERATOR_TYPE activation = DML_OPERATOR_INVALID;
        float param1 = 0.0f;
        float param2 = 0.0f;

        FusedActivation() = default;

        /* implicit */ FusedActivation(DML_OPERATOR_TYPE activation, float param1 = 0.0f, float param2 = 0.0f)
            : activation(activation), param1(param1), param2(param2)
        {}
    };

    // Implementation detail helper for determining if a list of expressions share the same GraphBuilder.
    namespace detail
    {
        inline bool HasSameOwner(Span<const Expression> exprs)
        {
            if (exprs.size() == 0)
            {
                return true;
            }

            detail::GraphBuilder* owner = exprs.begin()->Impl()->GetGraphBuilder();
            for (Expression expr : exprs)
            {
                if (expr.Impl()->GetGraphBuilder() != owner)
                {
                    return false;
                }
            }

            return true;
        }

        inline bool HasSameOwner(std::initializer_list<Expression> exprs)
        {
            Span<const Expression> span(exprs.begin(), exprs.size());
            return HasSameOwner(span);
        }

        inline bool HasSameDataType(Span<const Expression> exprs)
        {
            if (exprs.size() == 0)
            {
                return true;
            }

            DML_TENSOR_DATA_TYPE dataType = exprs.begin()->Impl()->GetOutputDesc().dataType;
            for (Expression expr : exprs)
            {
                if (expr.Impl()->GetOutputDesc().dataType != dataType)
                {
                    return false;
                }
            }

            return true;
        }

        inline bool HasSameDataType(std::initializer_list<Expression> exprs)
        {
            Span<const Expression> span(exprs.begin(), exprs.size());
            return HasSameDataType(span);
        }
    } // namespace detail

    // Expression implementation helpers
    namespace detail
    {
        template <DML_OPERATOR_TYPE OperatorType, typename TDesc>
        Expression ElementWiseUnary(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias)
        {
            detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

            TensorDesc inputTensor = input.Impl()->GetOutputDesc();
            TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetOutputLayout()); // Same as input

            TDesc desc = {};
            desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
            desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
            desc.ScaleBias = scaleBias ? &scaleBias.value() : nullptr;

            detail::NodeOutput* const inputs[] = { input.Impl() };
            detail::NodeID node = builder->CreateOperatorNode(OperatorType, &desc, inputs);
            detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

            return output;
        }

        template <DML_OPERATOR_TYPE OperatorType, typename TDesc>
        Expression ElementWiseUnary(Expression input, DML_TENSOR_DATA_TYPE outputDataType = DML_TENSOR_DATA_TYPE_UNKNOWN)
        {
            detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

            TensorDesc inputTensor = input.Impl()->GetOutputDesc();
         
            if (outputDataType == DML_TENSOR_DATA_TYPE_UNKNOWN)
            {
                outputDataType = inputTensor.dataType;
            }
            TensorDesc outputTensor(outputDataType, inputTensor.sizes, builder->GetOutputLayout());

            TDesc desc = {};
            desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
            desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

            detail::NodeOutput* const inputs[] = { input.Impl() };
            detail::NodeID node = builder->CreateOperatorNode(OperatorType, &desc, inputs);
            detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

            return output;
        }

        template <DML_OPERATOR_TYPE OperatorType, typename TDesc>
        Expression ElementWiseBinary(Expression a, Expression b)
        {
            assert(detail::HasSameOwner({ a, b }));
            detail::GraphBuilder* builder = a.Impl()->GetGraphBuilder();

            TensorDesc aTensor = a.Impl()->GetOutputDesc();
            TensorDesc bTensor = b.Impl()->GetOutputDesc();
            TensorDesc outputTensor(aTensor.dataType, aTensor.sizes, builder->GetOutputLayout()); // Same as input

            TDesc desc = {};
            desc.ATensor = aTensor.AsPtr<DML_TENSOR_DESC>();
            desc.BTensor = bTensor.AsPtr<DML_TENSOR_DESC>();
            desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

            detail::NodeOutput* const inputs[] = { a.Impl(), b.Impl() };
            detail::NodeID node = builder->CreateOperatorNode(OperatorType, &desc, inputs);
            detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

            return output;
        }

        // Used to reserve some space on the stack for setting up fused activation operator descs.
        struct FusedActivationStorage
        {
            DML_OPERATOR_DESC opDesc;

            // All fuseable activation descs have a common layout: two tensor desc pointers and up to 2 optional
            // float parameters, so just use LINEAR as an archetype
            DML_ACTIVATION_LINEAR_OPERATOR_DESC activationDesc;
        };

        // Returns the correct value for filling out fused activation fields in the DML API, e.g.
        // DML_CONVOLUTION_OPERATOR_DESC::FusedActivation. The descs themselves are stored in the `storage` outptr.
        inline const DML_OPERATOR_DESC* GetFusedActivationPtr(
            FusedActivation fusedActivation,
            _Out_ FusedActivationStorage* storage)
        {
            if (fusedActivation.activation == DML_OPERATOR_INVALID)
            {
                // No fused activation
                return nullptr;
            }

            storage->activationDesc.InputTensor = nullptr;
            storage->activationDesc.OutputTensor = nullptr;
            storage->activationDesc.Alpha = fusedActivation.param1;
            storage->activationDesc.Beta = fusedActivation.param2;
            
            storage->opDesc.Type = fusedActivation.activation;
            storage->opDesc.Desc = &storage->activationDesc;

            return &storage->opDesc;
        }

    } // namespace detail

    inline Expression InputTensor(Scope& scope, uint32_t inputIndex, TensorDesc desc)
    {
        detail::GraphBuilder* builder = scope.Impl();

        detail::NodeID node = builder->CreateInputNode(inputIndex);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(desc));
        return output;
    }

    inline Expression Identity(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_IDENTITY, DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Abs(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_ABS, DML_ELEMENT_WISE_ABS_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression ACos(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_ACOS, DML_ELEMENT_WISE_ACOS_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Add(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_ADD, DML_ELEMENT_WISE_ADD_OPERATOR_DESC>(a, b);
    }

    inline Expression BitAnd(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_BIT_AND, DML_ELEMENT_WISE_BIT_AND_OPERATOR_DESC>(a, b);
    }

    inline Expression BitOr(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_BIT_OR, DML_ELEMENT_WISE_BIT_OR_OPERATOR_DESC>(a, b);
    }

    inline Expression BitXor(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_BIT_XOR, DML_ELEMENT_WISE_BIT_XOR_OPERATOR_DESC>(a, b);
    }

    inline Expression BitShiftLeft(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_LEFT, DML_ELEMENT_WISE_BIT_SHIFT_LEFT_OPERATOR_DESC>(a, b);
    }

    inline Expression BitShiftRight(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_RIGHT, DML_ELEMENT_WISE_BIT_SHIFT_RIGHT_OPERATOR_DESC>(a, b);
    }

    inline Expression BitNot(Expression a)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_BIT_NOT, DML_ELEMENT_WISE_BIT_NOT_OPERATOR_DESC>(a);
    }

    inline Expression BitCount(Expression a, DML_TENSOR_DATA_TYPE outputDataType = DML_TENSOR_DATA_TYPE_UINT8)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_BIT_COUNT, DML_ELEMENT_WISE_BIT_COUNT_OPERATOR_DESC>(a, outputDataType);
    }

#if NTDDI_VERSION >= NTDDI_WIN10_VB
    inline Expression Add1(Expression a, Expression b, FusedActivation fusedActivation = {})
    {
        assert(detail::HasSameOwner({ a, b }));
        detail::GraphBuilder* builder = a.Impl()->GetGraphBuilder();

        TensorDesc aTensor = a.Impl()->GetOutputDesc();
        TensorDesc bTensor = b.Impl()->GetOutputDesc();
        TensorDesc outputTensor(aTensor.dataType, aTensor.sizes, builder->GetOutputLayout()); // Same as input
        detail::FusedActivationStorage storage;

        DML_ELEMENT_WISE_ADD1_OPERATOR_DESC desc = {};
        desc.ATensor = aTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BTensor = bTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.FusedActivation = detail::GetFusedActivationPtr(fusedActivation, &storage);

        detail::NodeOutput* const inputs[] = { a.Impl(), b.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_ADD1, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }
#endif

    inline Expression ASin(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_ASIN, DML_ELEMENT_WISE_ASIN_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression ATan(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_ATAN, DML_ELEMENT_WISE_ATAN_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Ceil(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_CEIL, DML_ELEMENT_WISE_CEIL_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Clip(Expression input, float min, float max, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetOutputLayout()); // Same as input

        DML_ELEMENT_WISE_CLIP_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleBias = scaleBias ? &scaleBias.value() : nullptr;
        desc.Min = min;
        desc.Max = max;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_CLIP, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Cos(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_COS, DML_ELEMENT_WISE_COS_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Divide(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_DIVIDE, DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC>(a, b);
    }

    inline Expression Erf(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_ERF, DML_ELEMENT_WISE_ERF_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Exp(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_EXP, DML_ELEMENT_WISE_EXP_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Floor(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_FLOOR, DML_ELEMENT_WISE_FLOOR_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Log(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_LOG, DML_ELEMENT_WISE_LOG_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression LogicalAnd(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND, DML_ELEMENT_WISE_LOGICAL_AND_OPERATOR_DESC>(a, b);
    }

    inline Expression LogicalEquals(Expression a, Expression b, DML_TENSOR_DATA_TYPE outputDataType = DML_TENSOR_DATA_TYPE_UINT8)
    {
        assert(detail::HasSameOwner({ a, b }));
        detail::GraphBuilder* builder = a.Impl()->GetGraphBuilder();

        TensorDesc aTensor = a.Impl()->GetOutputDesc();
        TensorDesc bTensor = b.Impl()->GetOutputDesc();
        TensorDesc outputTensor(outputDataType, aTensor.sizes, builder->GetOutputLayout());

        DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_DESC desc = {};
        desc.ATensor = aTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BTensor = bTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeOutput* const inputs[] = { a.Impl(), b.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression LogicalGreaterThan(Expression a, Expression b, DML_TENSOR_DATA_TYPE outputDataType = DML_TENSOR_DATA_TYPE_UINT8)
    {
        assert(detail::HasSameOwner({ a, b }));
        detail::GraphBuilder* builder = a.Impl()->GetGraphBuilder();

        TensorDesc aTensor = a.Impl()->GetOutputDesc();
        TensorDesc bTensor = b.Impl()->GetOutputDesc();
        TensorDesc outputTensor(outputDataType, aTensor.sizes, builder->GetOutputLayout());

        DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OPERATOR_DESC desc = {};
        desc.ATensor = aTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BTensor = bTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeOutput* const inputs[] = { a.Impl(), b.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression LogicalLessThan(Expression a, Expression b, DML_TENSOR_DATA_TYPE outputDataType = DML_TENSOR_DATA_TYPE_UINT8)
    {
        assert(detail::HasSameOwner({ a, b }));
        detail::GraphBuilder* builder = a.Impl()->GetGraphBuilder();

        TensorDesc aTensor = a.Impl()->GetOutputDesc();
        TensorDesc bTensor = b.Impl()->GetOutputDesc();
        TensorDesc outputTensor(outputDataType, aTensor.sizes, builder->GetOutputLayout());

        DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC desc = {};
        desc.ATensor = aTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BTensor = bTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeOutput* const inputs[] = { a.Impl(), b.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression LogicalNot(Expression input)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetOutputLayout()); // Same as input

        DML_ELEMENT_WISE_LOGICAL_NOT_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_LOGICAL_NOT, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression LogicalOr(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_LOGICAL_OR, DML_ELEMENT_WISE_LOGICAL_OR_OPERATOR_DESC>(a, b);
    }

    inline Expression LogicalXor(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_LOGICAL_XOR, DML_ELEMENT_WISE_LOGICAL_XOR_OPERATOR_DESC>(a, b);
    }

    inline Expression Max(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_MAX, DML_ELEMENT_WISE_MAX_OPERATOR_DESC>(a, b);
    }

    inline Expression Mean(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_MEAN, DML_ELEMENT_WISE_MEAN_OPERATOR_DESC>(a, b);
    }

    inline Expression Min(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_MIN, DML_ELEMENT_WISE_MIN_OPERATOR_DESC>(a, b);
    }

    inline Expression ModulusFloor(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_MODULUS_FLOOR, DML_ELEMENT_WISE_MODULUS_FLOOR_OPERATOR_DESC>(a, b);
    }

    inline Expression ModulusTruncate(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_MODULUS_TRUNCATE, DML_ELEMENT_WISE_MODULUS_TRUNCATE_OPERATOR_DESC>(a, b);
    }

    inline Expression Multiply(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_MULTIPLY, DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC>(a, b);
    }

    inline Expression Pow(Expression input, Expression exponent, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        assert(detail::HasSameOwner({ input, exponent }));
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc exponentTensor = exponent.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetOutputLayout()); // Same as input

        DML_ELEMENT_WISE_POW_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ExponentTensor = exponentTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleBias = scaleBias ? &scaleBias.value() : nullptr;

        detail::NodeOutput* const inputs[] = { input.Impl(), exponent.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_POW, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Pow(Expression input, float exponent, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetOutputLayout()); // Same as input

        DML_ELEMENT_WISE_CONSTANT_POW_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleBias = scaleBias ? &scaleBias.value() : nullptr;
        desc.Exponent = exponent;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_CONSTANT_POW, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Recip(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_RECIP, DML_ELEMENT_WISE_RECIP_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Sin(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_SIN, DML_ELEMENT_WISE_SIN_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Sqrt(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_SQRT, DML_ELEMENT_WISE_SQRT_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Subtract(Expression a, Expression b)
    {
        return detail::ElementWiseBinary<DML_OPERATOR_ELEMENT_WISE_SUBTRACT, DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC>(a, b);
    }

    inline Expression Tan(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_TAN, DML_ELEMENT_WISE_TAN_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Threshold(Expression input, float min, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetOutputLayout()); // Same as input

        DML_ELEMENT_WISE_THRESHOLD_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleBias = scaleBias ? &scaleBias.value() : nullptr;
        desc.Min = min;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_THRESHOLD, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression QuantizeLinear(Expression input, Expression scale, Expression zeroPoint)
    {
        assert(detail::HasSameOwner({ input, scale, zeroPoint }));

        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc scaleTensor = scale.Impl()->GetOutputDesc();
        TensorDesc zeroPointTensor = zeroPoint.Impl()->GetOutputDesc();
        TensorDesc outputTensor(DML_TENSOR_DATA_TYPE_UINT8, inputTensor.sizes, builder->GetOutputLayout());

        DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleTensor = scaleTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ZeroPointTensor = zeroPointTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeOutput* const inputs[] = { input.Impl(), scale.Impl(), zeroPoint.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression DequantizeLinear(Expression input, Expression scale, Expression zeroPoint)
    {
        assert(detail::HasSameOwner({ input, scale, zeroPoint }));

        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc scaleTensor = scale.Impl()->GetOutputDesc();
        TensorDesc zeroPointTensor = zeroPoint.Impl()->GetOutputDesc();
        TensorDesc outputTensor(DML_TENSOR_DATA_TYPE_FLOAT32, inputTensor.sizes, builder->GetOutputLayout());

        DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleTensor = scaleTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ZeroPointTensor = zeroPointTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeOutput* const inputs[] = { input.Impl(), scale.Impl(), zeroPoint.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Sinh(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_SINH, DML_ELEMENT_WISE_SINH_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Cosh(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_COSH, DML_ELEMENT_WISE_COSH_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Tanh(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_TANH, DML_ELEMENT_WISE_TANH_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression ASinh(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_ASINH, DML_ELEMENT_WISE_ASINH_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression ACosh(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_ACOSH, DML_ELEMENT_WISE_ACOSH_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression ATanh(Expression input, const Optional<DML_SCALE_BIAS>& scaleBias = NullOpt)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_ATANH, DML_ELEMENT_WISE_ATANH_OPERATOR_DESC>(input, scaleBias);
    }

    inline Expression Cast(Expression input, DML_TENSOR_DATA_TYPE targetDataType)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(targetDataType, inputTensor.sizes, builder->GetOutputLayout());

        DML_CAST_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_CAST, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Resample(
        Expression input,
        const TensorDesc::Dimensions& outputSizes,
        DML_INTERPOLATION_MODE mode,
        Span<const float> scales = {},
        Span<const float> inputPixelOffsets = std::array<float, 4>{0.5f, 0.5f, 0.5f, 0.5f},
        Span<const float> outputPixelOffsets = std::array<float, 4>{-0.5f, -0.5f, -0.5f, -0.5f})
    {
        assert(outputSizes.size() == 4);
        assert(inputPixelOffsets.size() == 4);
        assert(outputPixelOffsets.size() == 4);

        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, outputSizes, builder->GetOutputLayout());

        std::array<float, 4> dml_scales;

        if (scales.empty()) {
            dml_scales[0] = static_cast<float>(outputSizes[0]) / static_cast<float>(inputTensor.sizes[0]);
            dml_scales[1] = static_cast<float>(outputSizes[1]) / static_cast<float>(inputTensor.sizes[1]);
            dml_scales[2] = static_cast<float>(outputSizes[2]) / static_cast<float>(inputTensor.sizes[2]);
            dml_scales[3] = static_cast<float>(outputSizes[3]) / static_cast<float>(inputTensor.sizes[3]);
        } else {
            assert(scales.size() == 4);
            dml_scales[0] = scales[0];
            dml_scales[1] = scales[1];
            dml_scales[2] = scales[2];
            dml_scales[3] = scales[3];
        }

        DML_RESAMPLE1_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.InterpolationMode = mode;
        desc.DimensionCount = static_cast<UINT>(dml_scales.size());
        desc.Scales = dml_scales.data();
        desc.InputPixelOffsets = inputPixelOffsets.data();
        desc.OutputPixelOffsets = outputPixelOffsets.data();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_RESAMPLE1, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression ResampleGrad(
        Expression input,
        const TensorDesc::Dimensions& outputSizes,
        DML_INTERPOLATION_MODE mode,
        Span<const float> scales = {},
        Span<const float> inputPixelOffsets = std::array<float, 4>{-0.5f, -0.5f, -0.5f, -0.5f},
        Span<const float> outputPixelOffsets = std::array<float, 4>{0.5f, 0.5f, 0.5f, 0.5f})
    {
        assert(outputSizes.size() == 4);
        assert(inputPixelOffsets.size() == 4);
        assert(outputPixelOffsets.size() == 4);

        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, outputSizes, builder->GetOutputLayout());

        std::array<float, 4> dml_scales;

        if (scales.empty()) {
            dml_scales[0] = static_cast<float>(outputSizes[0]) / static_cast<float>(inputTensor.sizes[0]);
            dml_scales[1] = static_cast<float>(outputSizes[1]) / static_cast<float>(inputTensor.sizes[1]);
            dml_scales[2] = static_cast<float>(outputSizes[2]) / static_cast<float>(inputTensor.sizes[2]);
            dml_scales[3] = static_cast<float>(outputSizes[3]) / static_cast<float>(inputTensor.sizes[3]);
        } else {
            assert(scales.size() == 4);
            dml_scales[0] = scales[0];
            dml_scales[1] = scales[1];
            dml_scales[2] = scales[2];
            dml_scales[3] = scales[3];
        }

        DML_RESAMPLE_GRAD_OPERATOR_DESC desc = {};
        desc.InputGradientTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputGradientTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.InterpolationMode = mode;
        desc.DimensionCount = static_cast<UINT>(dml_scales.size());
        desc.Scales = dml_scales.data();
        desc.InputPixelOffsets = inputPixelOffsets.data();
        desc.OutputPixelOffsets = outputPixelOffsets.data();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_RESAMPLE_GRAD, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression If(Expression condition, Expression a, Expression b)
    {
        assert(detail::HasSameOwner({ condition, a, b }));
        assert(detail::HasSameDataType({ a, b }));

        detail::GraphBuilder* builder = condition.Impl()->GetGraphBuilder();

        TensorDesc conditionTensor = condition.Impl()->GetOutputDesc();
        assert(conditionTensor.dataType == DML_TENSOR_DATA_TYPE_UINT8);

        TensorDesc aTensor = a.Impl()->GetOutputDesc();
        TensorDesc bTensor = b.Impl()->GetOutputDesc();
        TensorDesc outputTensor(aTensor.dataType, aTensor.sizes, builder->GetOutputLayout());

        DML_ELEMENT_WISE_IF_OPERATOR_DESC desc = {};
        desc.ConditionTensor = conditionTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ATensor = aTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BTensor = bTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeOutput* const inputs[] = { condition.Impl(), a.Impl(), b.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_IF, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

#define DMLX_ACTIVATION_IMPL(_name) \
    do { \
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder(); \
        \
        TensorDesc inputTensor = input.Impl()->GetOutputDesc(); \
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetOutputLayout()); \
        \
        DML_##_name##_OPERATOR_DESC desc = {}; \
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>(); \
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>(); \
        \
        detail::NodeOutput* const inputs[] = { input.Impl() }; \
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_##_name, &desc, inputs); \
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor)); \
        \
        return output; \
    } while(0)
    
#define DMLX_ACTIVATION_IMPL_1(_name, _param1Name, _param1) \
    do { \
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder(); \
        \
        TensorDesc inputTensor = input.Impl()->GetOutputDesc(); \
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetOutputLayout()); \
        \
        DML_##_name##_OPERATOR_DESC desc = {}; \
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>(); \
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>(); \
        desc._param1Name = _param1; \
        \
        detail::NodeOutput* const inputs[] = { input.Impl() }; \
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_##_name, &desc, inputs); \
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor)); \
        \
        return output; \
    } while(0)
    
#define DMLX_ACTIVATION_IMPL_2(_name, _param1Name, _param1, _param2Name, _param2) \
    do { \
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder(); \
        \
        TensorDesc inputTensor = input.Impl()->GetOutputDesc(); \
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetOutputLayout()); \
        \
        DML_##_name##_OPERATOR_DESC desc = {}; \
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>(); \
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>(); \
        desc._param1Name = _param1; \
        desc._param2Name = _param2; \
        \
        detail::NodeOutput* const inputs[] = { input.Impl() }; \
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_##_name, &desc, inputs); \
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor)); \
        \
        return output; \
    } while(0)

    inline Expression ActivationElu(Expression input, float alpha)
    {
        DMLX_ACTIVATION_IMPL_1(ACTIVATION_ELU, Alpha, alpha);
    }

    inline Expression ActivationHardmax(Expression input)
    {
        DMLX_ACTIVATION_IMPL(ACTIVATION_HARDMAX);
    }

    inline Expression ActivationHardSigmoid(Expression input, float alpha, float beta)
    {
        DMLX_ACTIVATION_IMPL_2(ACTIVATION_HARD_SIGMOID, Alpha, alpha, Beta, beta);
    }

    inline Expression ActivationIdentity(Expression input)
    {
        DMLX_ACTIVATION_IMPL(ACTIVATION_IDENTITY);
    }

    inline Expression ActivationLeakyRelu(Expression input, float alpha)
    {
        DMLX_ACTIVATION_IMPL_1(ACTIVATION_LEAKY_RELU, Alpha, alpha);
    }

    inline Expression ActivationLinear(Expression input, float alpha, float beta)
    {
        DMLX_ACTIVATION_IMPL_2(ACTIVATION_LINEAR, Alpha, alpha, Beta, beta);
    }

    inline Expression ActivationLogSoftmax(Expression input)
    {
        DMLX_ACTIVATION_IMPL(ACTIVATION_LOG_SOFTMAX);
    }

    inline Expression ActivationParameterizedRelu(Expression input, Expression slope)
    {
        assert(detail::HasSameOwner({ input, slope }));

        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc slopeTensor = slope.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetOutputLayout());

        DML_ACTIVATION_PARAMETERIZED_RELU_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.SlopeTensor = slopeTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeOutput* const inputs[] = { input.Impl(), slope.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression ActivationParametricSoftplus(Expression input, float alpha, float beta)
    {
        DMLX_ACTIVATION_IMPL_2(ACTIVATION_PARAMETRIC_SOFTPLUS, Alpha, alpha, Beta, beta);
    }

    inline Expression ActivationRelu(Expression input)
    {
        DMLX_ACTIVATION_IMPL(ACTIVATION_RELU);
    }

    inline Expression ActivationScaledElu(Expression input, float alpha, float gamma)
    {
        DMLX_ACTIVATION_IMPL_2(ACTIVATION_SCALED_ELU, Alpha, alpha, Gamma, gamma);
    }

    inline Expression ActivationScaledTanh(Expression input, float alpha, float beta)
    {
        DMLX_ACTIVATION_IMPL_2(ACTIVATION_SCALED_TANH, Alpha, alpha, Beta, beta);
    }

    inline Expression ActivationSigmoid(Expression input)
    {
        DMLX_ACTIVATION_IMPL(ACTIVATION_SIGMOID);
    }

    inline Expression ActivationSoftmax(Expression input)
    {
        DMLX_ACTIVATION_IMPL(ACTIVATION_SOFTMAX);
    }

    inline Expression ActivationSoftplus(Expression input, float steepness = 1.0f)
    {
        DMLX_ACTIVATION_IMPL_1(ACTIVATION_SOFTPLUS, Steepness, steepness);
    }

    inline Expression ActivationSoftsign(Expression input)
    {
        DMLX_ACTIVATION_IMPL(ACTIVATION_SOFTSIGN);
    }

    inline Expression ActivationTanh(Expression input)
    {
        DMLX_ACTIVATION_IMPL(ACTIVATION_TANH);
    }

    inline Expression ActivationThresholdedRelu(Expression input, float alpha)
    {
        DMLX_ACTIVATION_IMPL_1(ACTIVATION_THRESHOLDED_RELU, Alpha, alpha);
    }

#undef DMLX_ACTIVATION_IMPL
#undef DMLX_ACTIVATION_IMPL_1
#undef DMLX_ACTIVATION_IMPL_2

    // Helper for setting parameters for the Convolution operator. Any unset parameters will be defaulted to the
    // following values:
    //   Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION
    //   Direction = DML_CONVOLUTION_DIRECTION_FORWARD
    //   Strides = { 1, 1 } for 2D convolution, { 1, 1, 1 } for 3D convolution
    //   Dilations = { 1, 1 } for 2D convolution, { 1, 1, 1 } for 3D convolution
    //   StartPadding = { 0, 0 } for 2D convolution, { 0, 0, 0 } for 3D convolution
    //   EndPadding = { 0, 0 } for 2D convolution, { 0, 0, 0 } for 3D convolution
    //   OutputPadding = { 0, 0 } for 2D convolution, { 0, 0, 0 } for 3D convolution
    //   GroupCount = 1
    //   FusedActivation = nullptr
    // 
    // This type is implicitly convertible to Expression, so it can be used in most contexts that require the
    // expression type.
    class ConvolutionExpression
    {
    public:
        ConvolutionExpression(Expression input, Expression filter, Optional<Expression> bias)
            : m_input(input), m_filter(filter), m_bias(bias)
        {}

        ConvolutionExpression& Mode(DML_CONVOLUTION_MODE mode) { m_mode = mode; return *this; }
        ConvolutionExpression& Direction(DML_CONVOLUTION_DIRECTION direction) { m_direction = direction; return *this; }
        ConvolutionExpression& Strides(Span<const uint32_t> strides) { m_strides.assign(strides.begin(), strides.end()); return *this; }
        ConvolutionExpression& Dilations(Span<const uint32_t> dilations) { m_dilations.assign(dilations.begin(), dilations.end()); return *this; }
        ConvolutionExpression& StartPadding(Span<const uint32_t> startPadding) { m_startPadding.assign(startPadding.begin(), startPadding.end()); return *this; }
        ConvolutionExpression& EndPadding(Span<const uint32_t> endPadding) { m_endPadding.assign(endPadding.begin(), endPadding.end()); return *this; }
        ConvolutionExpression& OutputPadding(Span<const uint32_t> outputPadding) { m_outputPadding.assign(outputPadding.begin(), outputPadding.end()); return *this; }
        ConvolutionExpression& OutputShape(Span<const uint32_t> outputShape) { m_outputShape.assign(outputShape.begin(), outputShape.end()); return *this; }
        ConvolutionExpression& GroupCount(uint32_t groupCount) { m_groupCount = groupCount; return *this; }
        ConvolutionExpression& FusedActivation(DML_OPERATOR_TYPE activation, float param1 = 0.0f, float param2 = 0.0f)
        {
            m_fusedActivation.activation = activation;
            m_fusedActivation.param1 = param1;
            m_fusedActivation.param2 = param2;
            return *this;
        }

        /* implicit */ operator Expression() const
        {
            assert(detail::HasSameOwner({ m_input, m_filter }));
            assert(!m_bias || detail::HasSameOwner({ m_input, *m_bias }));

            detail::GraphBuilder* builder = m_input.Impl()->GetGraphBuilder();

            TensorDesc inputTensor = m_input.Impl()->GetOutputDesc();
            TensorDesc filterTensor = m_filter.Impl()->GetOutputDesc();
            TensorDesc biasTensor;
            if (m_bias)
            {
                biasTensor = m_bias->Impl()->GetOutputDesc();
            }

            uint32_t dimensionCount = static_cast<uint32_t>(inputTensor.sizes.size());

            assert(dimensionCount == 4 || dimensionCount == 5);
            uint32_t spatialDimensionCount = dimensionCount - 2;

            // If the spatial dimension count is 2, we'll just use the first two elements by setting
            // DimensionCount = 2 in the desc
            const uint32_t defaultStridesAndDilations[3] = { 1, 1, 1 };
            const uint32_t defaultPadding[3] = { 0, 0, 0 };

            assert(m_strides.empty() || m_strides.size() == spatialDimensionCount);
            assert(m_dilations.empty() || m_dilations.size() == spatialDimensionCount);
            assert(m_startPadding.empty() || m_startPadding.size() == spatialDimensionCount);
            assert(m_endPadding.empty() || m_endPadding.size() == spatialDimensionCount);
            assert(m_outputPadding.empty() || m_outputPadding.size() == spatialDimensionCount);
            assert(m_outputShape.empty() || m_outputShape.size() == inputTensor.sizes.size());

            Span<const uint32_t> strides = m_strides.empty() ? Span<const uint32_t>{ defaultStridesAndDilations } : m_strides;
            Span<const uint32_t> dilations = m_dilations.empty() ? Span<const uint32_t>{ defaultStridesAndDilations } : m_dilations;
            Span<const uint32_t> startPadding = m_startPadding.empty() ? Span<const uint32_t>{ defaultPadding } : m_startPadding;
            Span<const uint32_t> endPadding = m_endPadding.empty() ? Span<const uint32_t>{ defaultPadding } : m_endPadding;
            Span<const uint32_t> outputPadding = m_outputPadding.empty() ? Span<const uint32_t>{ defaultPadding } : m_outputPadding;

            // Compute the output shapes

            TensorDesc::Dimensions outputSizes;

            if (!m_outputShape.empty())
            {
                outputSizes = m_outputShape;
            }
            else if (m_direction == DML_CONVOLUTION_DIRECTION_FORWARD)
            {
                outputSizes.push_back(inputTensor.sizes[0]); // output[N] = input[N]
                outputSizes.push_back(filterTensor.sizes[0]); // output[C] = filter[N]

                for (uint32_t dim = 0; dim < spatialDimensionCount; ++dim)
                {
                    uint32_t inputSize = inputTensor.sizes[dim + 2];
                    uint32_t paddedSize = inputSize + startPadding[dim] + endPadding[dim];

                    uint32_t windowSize = filterTensor.sizes[dim + 2];
                    uint32_t kernelSize = 1 + (windowSize - 1) * dilations[dim];

                    assert(kernelSize <= paddedSize);
                    assert(strides[dim] != 0);

                    outputSizes.push_back(1 + (paddedSize - kernelSize) / strides[dim]);
                }
            }
            else if (m_direction == DML_CONVOLUTION_DIRECTION_BACKWARD)
            {
                // TODO: implement me
                assert(false);
            }
            else
            {
                DMLX_THROW(E_UNEXPECTED);
            }

            TensorDesc outputTensor(inputTensor.dataType, std::move(outputSizes), builder->GetOutputLayout());
            detail::FusedActivationStorage storage;

            DML_CONVOLUTION_OPERATOR_DESC desc = {};
            desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
            desc.FilterTensor = filterTensor.AsPtr<DML_TENSOR_DESC>();
            desc.BiasTensor = m_bias ? biasTensor.AsPtr<DML_TENSOR_DESC>() : nullptr;
            desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
            desc.Mode = m_mode;
            desc.Direction = m_direction;
            desc.DimensionCount = spatialDimensionCount;
            desc.Strides = strides.data();
            desc.Dilations = dilations.data();
            desc.StartPadding = startPadding.data();
            desc.EndPadding = endPadding.data();
            desc.OutputPadding = outputPadding.data();
            desc.GroupCount = m_groupCount;
            desc.FusedActivation = detail::GetFusedActivationPtr(m_fusedActivation, &storage);

            SmallVector<detail::NodeOutput*, 3> inputs = { m_input.Impl(), m_filter.Impl() };
            if (m_bias)
            {
                inputs.push_back(m_bias->Impl());
            }

            detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_CONVOLUTION, &desc, inputs);
            detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

            return output;
        }

    private:
        Expression m_input;
        Expression m_filter;
        Optional<Expression> m_bias;
        DML_CONVOLUTION_MODE m_mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
        DML_CONVOLUTION_DIRECTION m_direction = DML_CONVOLUTION_DIRECTION_FORWARD;
        SmallVector<uint32_t, 3> m_strides = {};
        SmallVector<uint32_t, 3> m_dilations = {};
        SmallVector<uint32_t, 3> m_startPadding = {};
        SmallVector<uint32_t, 3> m_endPadding = {};
        SmallVector<uint32_t, 3> m_outputPadding = {};
        TensorDesc::Dimensions m_outputShape = {};
        uint32_t m_groupCount = 1;
        dml::FusedActivation m_fusedActivation;
    };

    inline ConvolutionExpression Convolution(
        Expression input,
        Expression filter,
        Optional<Expression> bias = NullOpt,
        DML_CONVOLUTION_MODE mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION,
        DML_CONVOLUTION_DIRECTION direction = DML_CONVOLUTION_DIRECTION_FORWARD,
        Span<const uint32_t> strides = {},
        Span<const uint32_t> dilations = {},
        Span<const uint32_t> startPadding = {},
        Span<const uint32_t> endPadding = {},
        Span<const uint32_t> outputPadding = {},
        Span<const uint32_t> outputShape = {},
        uint32_t groupCount =  1,
        DML_OPERATOR_TYPE fusedActivation = DML_OPERATOR_INVALID,
        float fusedActivationParam1 = 0.0f,
        float fusedActivationParam2 = 0.0f)
    {
        return ConvolutionExpression(input, filter, bias)
            .Mode(mode)
            .Direction(direction)
            .Strides(strides)
            .Dilations(dilations)
            .StartPadding(startPadding)
            .EndPadding(endPadding)
            .OutputPadding(outputPadding)
            .OutputShape(outputShape)
            .GroupCount(groupCount)
            .FusedActivation(fusedActivation, fusedActivationParam1, fusedActivationParam2);
    }
    
    // Helper for setting parameters for the GEMM operator. Any unset parameters will be defaulted to the
    // following values:
    //   TransA = DML_MATRIX_TRANSFORM_NONE
    //   TransB = DML_MATRIX_TRANSFORM_NONE
    //   Alpha = 1.0f
    //   Beta = 1.0f
    //   FusedActivation = nullptr
    // 
    // This type is implicitly convertible to Expression, so it can be used in most contexts that require the
    // expression type.
    class GemmExpression
    {
    public:
        GemmExpression(Expression a, Expression b, Optional<Expression> c)
            : m_a(a), m_b(b), m_c(c)
        {}

        GemmExpression& TransA(DML_MATRIX_TRANSFORM transA) { m_transA = transA; return *this; }
        GemmExpression& TransB(DML_MATRIX_TRANSFORM transB) { m_transB = transB; return *this; }
        GemmExpression& Alpha(float alpha) { m_alpha = alpha; return *this; }
        GemmExpression& Beta(float beta) { m_beta = beta; return *this; }
        GemmExpression& FusedActivation(DML_OPERATOR_TYPE activation, float param1 = 0.0f, float param2 = 0.0f)
        {
            m_fusedActivation.activation = activation;
            m_fusedActivation.param1 = param1;
            m_fusedActivation.param2 = param2;
            return *this;
        }

        /* implicit */ operator Expression() const
        {
            assert(detail::HasSameOwner({ m_a, m_b }));
            assert(!m_c || detail::HasSameOwner({ m_a, *m_c }));

            detail::GraphBuilder* builder = m_a.Impl()->GetGraphBuilder();

            TensorDesc aTensor = m_a.Impl()->GetOutputDesc();
            TensorDesc bTensor = m_b.Impl()->GetOutputDesc();
            TensorDesc cTensor;
            if (m_c)
            {
                cTensor = m_c->Impl()->GetOutputDesc();
            }

            TensorDesc::Dimensions outputSizes;
            outputSizes.push_back(aTensor.sizes[0]); // output[N] = input[N]
            outputSizes.push_back(aTensor.sizes[1]); // output[C] = input[C]
            outputSizes.push_back(m_transA == DML_MATRIX_TRANSFORM_NONE ? aTensor.sizes[2] : aTensor.sizes[3]);
            outputSizes.push_back(m_transB == DML_MATRIX_TRANSFORM_NONE ? bTensor.sizes[3] : bTensor.sizes[2]);

            TensorDesc outputTensor(aTensor.dataType, std::move(outputSizes), builder->GetOutputLayout());
            detail::FusedActivationStorage storage;

            DML_GEMM_OPERATOR_DESC desc = {};
            desc.ATensor = aTensor.AsPtr<DML_TENSOR_DESC>();
            desc.BTensor = bTensor.AsPtr<DML_TENSOR_DESC>();
            desc.CTensor = m_c ? cTensor.AsPtr<DML_TENSOR_DESC>() : nullptr;
            desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
            desc.TransA = m_transA;
            desc.TransB = m_transB;
            desc.Alpha = m_alpha;
            desc.Beta = m_beta;
            desc.FusedActivation = detail::GetFusedActivationPtr(m_fusedActivation, &storage);

            SmallVector<detail::NodeOutput*, 3> inputs = { m_a.Impl(), m_b.Impl() };
            if (m_c)
            {
                inputs.push_back(m_c->Impl());
            }

            detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_GEMM, &desc, inputs);
            detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

            return output;
        }

    private:
        Expression m_a;
        Expression m_b;
        Optional<Expression> m_c;
        DML_MATRIX_TRANSFORM m_transA = DML_MATRIX_TRANSFORM_NONE;
        DML_MATRIX_TRANSFORM m_transB = DML_MATRIX_TRANSFORM_NONE;
        float m_alpha = 1.0f;
        float m_beta = 1.0f;
        dml::FusedActivation m_fusedActivation;
    };

    inline GemmExpression Gemm(
        Expression a,
        Expression b,
        Optional<Expression> c = NullOpt,
        DML_MATRIX_TRANSFORM transA = DML_MATRIX_TRANSFORM_NONE,
        DML_MATRIX_TRANSFORM transB = DML_MATRIX_TRANSFORM_NONE,
        float alpha = 1.0f,
        float beta = 1.0f,
        DML_OPERATOR_TYPE fusedActivation = DML_OPERATOR_INVALID,
        float fusedActivationParam1 = 0.0f,
        float fusedActivationParam2 = 0.0f)
    {
        return GemmExpression(a, b, c)
            .TransA(transA)
            .TransB(transB)
            .Alpha(alpha)
            .Beta(beta)
            .FusedActivation(fusedActivation, fusedActivationParam1, fusedActivationParam2);
    }

    // If `axes` is not specified, by default this reduces the entire tensor to single element.
    inline Expression Reduce(Expression input, DML_REDUCE_FUNCTION function, Span<const uint32_t> axes = {})
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        const uint32_t allAxes[] = { 0, 1, 2, 3 };
        if (axes.empty())
        {
            axes = allAxes;
        }

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();

        // Compute the output tensor dimensions
        TensorDesc::Dimensions outputSizes;
        for (uint32_t i = 0; i < 4; ++i)
        {
            // If the dimension is to be reduced, this dimension in the output tensor has a size of 1, otherwise
            // it matches the input tensor.
            const bool dimensionIsReduced = std::find(axes.begin(), axes.end(), i) != axes.end();
            if (dimensionIsReduced)
            {
                outputSizes.push_back(1);
            }
            else
            {
                outputSizes.push_back(inputTensor.sizes[i]);
            }
        }

        // ARGMIN and ARGMAX reduction produce a UINT32 output; all other reductions produce an output with the same
        // type as the input.
        DML_TENSOR_DATA_TYPE outputDataType;
        if (function == DML_REDUCE_FUNCTION_ARGMIN || function == DML_REDUCE_FUNCTION_ARGMAX)
        {
            outputDataType = DML_TENSOR_DATA_TYPE_UINT32;
        }
        else
        {
            outputDataType = inputTensor.dataType;
        }

        TensorDesc outputTensor(outputDataType, outputSizes, builder->GetOutputLayout());

        DML_REDUCE_OPERATOR_DESC desc = {};
        desc.Function = function;
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.AxisCount = static_cast<uint32_t>(axes.size());
        desc.Axes = axes.data();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_REDUCE, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Upsample2D(Expression input, DML_SIZE_2D scaleSize, DML_INTERPOLATION_MODE interpolationMode)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();

        TensorDesc::Dimensions outputSizes;
        outputSizes.push_back(inputTensor.sizes[0]);                    // output[N] = input[N]
        outputSizes.push_back(inputTensor.sizes[1]);                    // output[C] = input[C]
        outputSizes.push_back(inputTensor.sizes[2] * scaleSize.Height); // output[H] = input[H] * scaleH
        outputSizes.push_back(inputTensor.sizes[3] * scaleSize.Width);  // output[W] = input[W] * scaleW
        TensorDesc outputTensor(inputTensor.dataType, outputSizes, builder->GetOutputLayout());

        DML_UPSAMPLE_2D_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleSize = scaleSize;
        desc.InterpolationMode = interpolationMode;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_UPSAMPLE_2D, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression BatchNormalization(
        Expression input,
        Expression mean,
        Expression variance,
        Expression scale,
        Expression bias,
        bool spatial,
        float epsilon, 
        DML_OPERATOR_TYPE fusedActivation = DML_OPERATOR_INVALID,
        float fusedActivationParam1 = 0.0f,
        float fusedActivationParam2 = 0.0f)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc meanTensor = mean.Impl()->GetOutputDesc();
        TensorDesc varianceTensor = variance.Impl()->GetOutputDesc();
        TensorDesc scaleTensor = scale.Impl()->GetOutputDesc();
        TensorDesc biasTensor = bias.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetOutputLayout());

        detail::FusedActivationStorage storage;
        dml::FusedActivation fusedActivationData(fusedActivation, fusedActivationParam1, fusedActivationParam2);

        DML_BATCH_NORMALIZATION_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.MeanTensor = meanTensor.AsPtr<DML_TENSOR_DESC>();
        desc.VarianceTensor = varianceTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleTensor = scaleTensor.AsPtr<DML_TENSOR_DESC>();
        desc.BiasTensor = biasTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Spatial = spatial;
        desc.Epsilon = epsilon;
        desc.FusedActivation = detail::GetFusedActivationPtr(fusedActivationData, &storage);

        detail::NodeOutput* const inputs[] = { input.Impl(), mean.Impl(), variance.Impl(), scale.Impl(), bias.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_BATCH_NORMALIZATION, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline std::vector<Expression> Split(
        Expression input,
        uint32_t axis,
        Span<const uint32_t> outputAxisSizes)
    {
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        uint32_t axisSizeSum = 0;

        std::vector<TensorDesc> outputTensors;
        outputTensors.reserve(outputAxisSizes.size());

        std::vector<DML_TENSOR_DESC> outputDescs;
        outputDescs.reserve(outputAxisSizes.size());

        for (uint32_t outputAxisSize : outputAxisSizes)
        {
            TensorDesc::Dimensions outputSizes = inputTensor.sizes;
            outputSizes[axis] = outputAxisSize;

            TensorDesc tensorDesc(inputTensor.dataType, outputSizes, builder->GetOutputLayout());
            outputTensors.push_back(std::move(tensorDesc));
            outputDescs.push_back(*outputTensors.back().AsPtr<DML_TENSOR_DESC>());

            axisSizeSum += outputAxisSize;
        }

        if (axisSizeSum != inputTensor.sizes[axis])
        {
            DMLX_THROW(E_UNEXPECTED);
        }

        DML_SPLIT_OPERATOR_DESC desc = {};
        desc.Axis = axis;
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensors = outputDescs.data();
        desc.OutputCount = static_cast<uint32_t>(outputAxisSizes.size());

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_SPLIT, &desc, inputs);

        std::vector<Expression> outputs;
        outputs.reserve(outputAxisSizes.size());

        for (uint32_t i = 0; i < outputAxisSizes.size(); ++i)
        {
            outputs.push_back(builder->CreateNodeOutput(node, i, std::move(outputTensors[i])));
        }

        return outputs;
    }

    inline Expression Join(
        Span<const Expression> inputs,
        uint32_t axis)
    {
        if (inputs.empty())
        {
            DMLX_THROW(E_UNEXPECTED);
        }

        detail::GraphBuilder* builder = inputs[0].Impl()->GetGraphBuilder();
        DML_TENSOR_DATA_TYPE dataType = inputs[0].Impl()->GetOutputDesc().dataType;

        TensorDesc::Dimensions outputSizes = inputs[0].Impl()->GetOutputDesc().sizes;
        outputSizes[axis] = 0;

        std::vector<TensorDesc> inputTensors;
        inputTensors.reserve(inputs.size());

        std::vector<DML_TENSOR_DESC> inputDescs;
        inputDescs.reserve(inputs.size());

        std::vector<detail::NodeOutput*> inputNodes;
        inputNodes.reserve(inputs.size());

        for (Expression input : inputs)
        {
            inputTensors.push_back(input.Impl()->GetOutputDesc());
            TensorDesc& inputTensor = inputTensors.back();
            outputSizes[axis] += inputTensor.sizes[axis];
            inputDescs.push_back(*inputTensor.AsPtr<DML_TENSOR_DESC>());
            inputNodes.push_back(input.Impl());
        }

        TensorDesc outputTensor(dataType, outputSizes, builder->GetOutputLayout());

        DML_JOIN_OPERATOR_DESC desc = {};
        desc.Axis = axis;
        desc.InputCount = static_cast<uint32_t>(inputDescs.size());
        desc.InputTensors = inputDescs.data();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_JOIN, &desc, inputNodes);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Gather(
        Expression input,
        Expression indices,
        uint32_t axis,
        uint32_t indexDimensions,
        Optional<TensorDesc::Dimensions> outputStrides)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc indicesTensor = indices.Impl()->GetOutputDesc();

        TensorDesc::Dimensions outputSizes = { 1, 1, 1, 1 };

        // All dimensions after the axis should be the same as the input
        int outputDim = 3;
        for (; static_cast<uint32_t>(outputDim) > axis; --outputDim)
        {
            outputSizes[outputDim] = inputTensor.sizes[outputDim];
        }

        // All dimensions within the range [axis - indexDimensions, axis] should be the same as the indices
        int indexDim = 3;
        for (; outputDim > static_cast<int>(axis) - static_cast<int>(indexDimensions); --outputDim, --indexDim)
        {
            outputSizes[outputDim] = indicesTensor.sizes[indexDim];
        }

        // All dimensions before axis - indexDimensions should be the same as the input
        int inputDim = axis - 1;
        for (; outputDim >= 0 && inputDim >= 0; --outputDim, --inputDim)
        {
            outputSizes[outputDim] = inputTensor.sizes[inputDim];
        }

        TensorDesc outputTensor(inputTensor.dataType, outputSizes, outputStrides);

        DML_GATHER_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.IndicesTensor = indicesTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Axis = axis;
        desc.IndexDimensions = indexDimensions;

        detail::NodeOutput* const inputs[] = { input.Impl(), indices.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_GATHER, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression GatherElements(
        Expression input,
        Expression indices,
        uint32_t axis,
        Optional<TensorDesc::Dimensions> outputStrides)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc indicesTensor = indices.Impl()->GetOutputDesc();

        TensorDesc outputTensor(inputTensor.dataType, indicesTensor.sizes, outputStrides);

        DML_GATHER_ELEMENTS_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.IndicesTensor = indicesTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Axis = axis;

        detail::NodeOutput* const inputs[] = { input.Impl(), indices.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_GATHER_ELEMENTS, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression OneHot(
        Expression indices, 
        Expression values, 
        uint32_t axis,
        TensorDesc outputTensor)
    {
        detail::GraphBuilder* builder = indices.Impl()->GetGraphBuilder();
        TensorDesc indicesTensor = indices.Impl()->GetOutputDesc();
        TensorDesc valuesTensor = values.Impl()->GetOutputDesc();


        DML_ONE_HOT_OPERATOR_DESC desc = {};
        desc.IndicesTensor = indicesTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ValuesTensor = valuesTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Axis = axis; 

        detail::NodeOutput* const inputs[] = { indices.Impl(), values.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ONE_HOT, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression FillValueConstant(
        Scope& scope, 
        DML_SCALAR_UNION start,
        TensorDesc outputDesc)
    {
        detail::GraphBuilder* builder = scope.Impl();

        DML_FILL_VALUE_CONSTANT_OPERATOR_DESC desc = {};
        desc.OutputTensor = outputDesc.AsPtr<DML_TENSOR_DESC>();
        desc.ValueDataType = outputDesc.dataType;
        desc.Value = start;

        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_FILL_VALUE_CONSTANT, &desc, {});
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputDesc));

        return output;
    }

    inline Expression FillValueSequence(
        Scope& scope, 
        DML_SCALAR_UNION start,
        DML_SCALAR_UNION delta, 
        TensorDesc outputDesc)
    {
        detail::GraphBuilder* builder = scope.Impl();

        DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC desc = {};
        desc.OutputTensor = outputDesc.AsPtr<DML_TENSOR_DESC>();
        desc.ValueDataType = outputDesc.dataType;
        desc.ValueStart = start;
        desc.ValueDelta = delta;

        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_FILL_VALUE_SEQUENCE, &desc, {});
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputDesc));

        return output;
    }

    inline std::vector<Expression> RandomGenerator(
        Expression inputState,
        uint32_t outputElementCount,
        bool outputState = true,
        DML_RANDOM_GENERATOR_TYPE type = DML_RANDOM_GENERATOR_TYPE_PHILOX_4X32_10)
    {
        detail::GraphBuilder* builder = inputState.Impl()->GetGraphBuilder();

        TensorDesc inputStateTensorDesc = inputState.Impl()->GetOutputDesc();
        TensorDesc outputTensorDesc = {};
        outputTensorDesc.sizes = { 1,1,1,outputElementCount };
        outputTensorDesc.dataType = DML_TENSOR_DATA_TYPE_UINT32;
        outputTensorDesc.totalTensorSizeInBytes = outputElementCount * sizeof(uint32_t);

        DML_RANDOM_GENERATOR_OPERATOR_DESC desc = {};
        desc.Type = type;
        desc.InputStateTensor = inputStateTensorDesc.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensorDesc.AsPtr<DML_TENSOR_DESC>();
        if (outputState)
        {
            // Input and output state have the same TensorDesc.
            desc.OutputStateTensor = inputStateTensorDesc.AsPtr<DML_TENSOR_DESC>();
        }

        detail::NodeOutput* const inputs[] = { inputState.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_RANDOM_GENERATOR, &desc, inputs);
        detail::NodeOutput* outOutput = builder->CreateNodeOutput(node, 0, std::move(outputTensorDesc));

        if (outputState)
        {
            TensorDesc outputStateTensorDesc = inputStateTensorDesc;
            detail::NodeOutput* outOutputState = builder->CreateNodeOutput(node, 1, std::move(outputStateTensorDesc));
            return { outOutput, outOutputState };
        }
        else
        {
            return { outOutput };
        }
    }

    inline Expression IsInfinity(Expression input, DML_IS_INFINITY_MODE infinityMode = DML_IS_INFINITY_MODE_EITHER)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(DML_TENSOR_DATA_TYPE_UINT8, inputTensor.sizes, builder->GetOutputLayout()); // Same as input

        DML_ELEMENT_WISE_IS_INFINITY_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.InfinityMode = infinityMode;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_IS_INFINITY, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression IsNaN(Expression input)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(DML_TENSOR_DATA_TYPE_UINT8, inputTensor.sizes, builder->GetOutputLayout()); // Same as input

        DML_ELEMENT_WISE_IS_NAN_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_IS_NAN, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Sign(Expression a)
    {
        return detail::ElementWiseUnary<DML_OPERATOR_ELEMENT_WISE_SIGN, DML_ELEMENT_WISE_SIGN_OPERATOR_DESC>(a);
    }

    // Helper for setting parameters for the MaxPooling operator. Any unset parameters will be defaulted to the
    // following values:
    //   OutputIndices = False
    //   Strides = 1 for each spatial dimension
    //   StartPadding = 0 for each spatial dimension
    //   EndPadding = 0 for each spatial dimension
    //   Dilations = 1 for each spatial dimension
    // 
    // This type is implicitly convertible to Expression, so it can be used in most contexts that require the
    // expression type.
    class MaxPoolingExpression
    {
    public:
        MaxPoolingExpression(Expression input, Span<const uint32_t> windowSize)
            : m_input(input), m_windowSize(windowSize)
        {}

        MaxPoolingExpression& OutputIndices(bool outputIndices) { m_outputIndices = outputIndices; return *this; }
        MaxPoolingExpression& Strides(Span<const uint32_t> strides) { m_strides.assign(strides.begin(), strides.end()); return *this; }
        MaxPoolingExpression& StartPadding(Span<const uint32_t> startPadding) { m_startPadding.assign(startPadding.begin(), startPadding.end()); return *this; }
        MaxPoolingExpression& EndPadding(Span<const uint32_t> endPadding) { m_endPadding.assign(endPadding.begin(), endPadding.end()); return *this; }
        MaxPoolingExpression& Dilations(Span<const uint32_t> dilations) { m_dilations.assign(dilations.begin(), dilations.end()); return *this; }

        operator std::vector<Expression>() const
        {
            detail::GraphBuilder* builder = m_input.Impl()->GetGraphBuilder();

            TensorDesc inputTensor = m_input.Impl()->GetOutputDesc();

            // If the spatial dimension count is 2, we'll just use the first two elements by setting
            // DimensionCount = 2 in the desc
            const uint32_t defaultStridesAndDilations[3] = { 1, 1, 1 };
            const uint32_t defaultPadding[3] = { 0, 0, 0 };

            assert(m_windowSize.size() == 2 || m_windowSize.size() == 3);
            assert(m_strides.empty() || m_strides.size() == m_windowSize.size());
            assert(m_dilations.empty() || m_dilations.size() == m_windowSize.size());
            assert(m_startPadding.empty() || m_startPadding.size() == m_windowSize.size());
            assert(m_endPadding.empty() || m_endPadding.size() == m_windowSize.size());

            Span<const uint32_t> strides = m_strides.empty() ? Span<const uint32_t>{ defaultStridesAndDilations } : m_strides;
            Span<const uint32_t> dilations = m_dilations.empty() ? Span<const uint32_t>{ defaultStridesAndDilations } : m_dilations;
            Span<const uint32_t> startPadding = m_startPadding.empty() ? Span<const uint32_t>{ defaultPadding } : m_startPadding;
            Span<const uint32_t> endPadding = m_endPadding.empty() ? Span<const uint32_t>{ defaultPadding } : m_endPadding;

            // Calculate output size
            TensorDesc::Dimensions outputSizes;
            outputSizes.push_back(inputTensor.sizes[0]); // N
            outputSizes.push_back(inputTensor.sizes[1]); // C
            for (size_t i = 0; i < m_windowSize.size(); i++)
            {
                uint32_t paddedInputSize = inputTensor.sizes[2 + i] + startPadding[i] + endPadding[i];
                uint32_t dilatedWindowSize = 1 + (m_windowSize[i] - 1) * dilations[i];
                uint32_t outputSize = (dilatedWindowSize >= paddedInputSize) ? 1 : (paddedInputSize - dilatedWindowSize) / strides[i] + 1;
                outputSizes.push_back(outputSize);
            }

            TensorDesc outputTensor(inputTensor.dataType, outputSizes, builder->GetOutputLayout());
            TensorDesc outputIndicesTensor(DML_TENSOR_DATA_TYPE_UINT32, outputSizes, builder->GetOutputLayout());

            assert(m_dilations.size() == 0);
            DML_MAX_POOLING1_OPERATOR_DESC desc = {};
            desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
            desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
            desc.OutputIndicesTensor = m_outputIndices ? outputIndicesTensor.AsPtr<DML_TENSOR_DESC>() : nullptr;
            desc.DimensionCount = static_cast<uint32_t>(m_windowSize.size());
            desc.Strides = strides.data();
            desc.WindowSize = m_windowSize.data();
            desc.StartPadding = startPadding.data();
            desc.EndPadding = endPadding.data();

            // Use DML_MAX_POOLING2 when implemented; for now, dilations are not allowed!
            assert(m_dilations.size() == 0); // desc.Dilations = dilations.data();

            detail::NodeOutput* const inputs[] = { m_input.Impl() };
            detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_MAX_POOLING1, &desc, inputs);

            detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));
            if (m_outputIndices)
            {
                detail::NodeOutput* outputIndices = builder->CreateNodeOutput(node, 1, std::move(outputIndicesTensor));
                return { output, outputIndices };
            }
            return { output };
        }

    private:
        Expression m_input;
        Span<const uint32_t> m_windowSize = {};
        SmallVector<uint32_t, 3> m_strides = {};
        SmallVector<uint32_t, 3> m_startPadding = {};
        SmallVector<uint32_t, 3> m_endPadding = {};
        SmallVector<uint32_t, 3> m_dilations = {};
        bool m_outputIndices = false;
    };

    inline MaxPoolingExpression MaxPooling(
        Expression input,
        Span<const uint32_t> windowSize,
        Span<const uint32_t> strides = {},
        Span<const uint32_t> startPadding = {},
        Span<const uint32_t> endPadding = {},
        Span<const uint32_t> dilations = {},
        bool outputIndices = false)
    {
        return MaxPoolingExpression(input, windowSize)
            .Strides(strides)
            .StartPadding(startPadding)
            .EndPadding(endPadding)
            .Dilations(dilations)
            .OutputIndices(outputIndices);
    }

    inline Expression Padding(
        Expression input, 
        DML_PADDING_MODE paddingMode, 
        float paddingValue, 
        Span<const uint32_t> startPadding,
        Span<const uint32_t> endPadding)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc::Dimensions outputSizes = inputTensor.sizes;

        assert(outputSizes.size() == startPadding.size());
        assert(outputSizes.size() == endPadding.size());

        for (size_t i = 0; i < outputSizes.size(); i++)
        {
            outputSizes[i] += startPadding[i] + endPadding[i];
        }
       
        TensorDesc outputTensor(inputTensor.dataType, outputSizes, builder->GetOutputLayout());

        DML_PADDING_OPERATOR_DESC paddingDesc = {};
        paddingDesc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        paddingDesc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        paddingDesc.PaddingMode = paddingMode;
        paddingDesc.PaddingValue = paddingValue;
        paddingDesc.DimensionCount = static_cast<uint32_t>(startPadding.size());
        paddingDesc.StartPadding = startPadding.data();
        paddingDesc.EndPadding = endPadding.data();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_PADDING, &paddingDesc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Slice1(
        Expression input,
        Span<const uint32_t> inputWindowOffsets,
        Span<const uint32_t> inputWindowSizes,
        Span<const int32_t> inputWindowStrides)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc::Dimensions outputSizes(inputTensor.sizes);

        assert(inputWindowOffsets.size() == outputSizes.size());
        assert(inputWindowOffsets.size() == inputWindowStrides.size());
        assert(inputWindowOffsets.size() == inputWindowSizes.size());

        for (size_t i = 0; i < outputSizes.size(); i++)
        {
            uint32_t minimumInputSize = (inputWindowSizes[i] - 1) / abs(inputWindowStrides[i]) + 1;
            outputSizes[i] = minimumInputSize;
        }

        TensorDesc outputTensor(inputTensor.dataType, outputSizes, builder->GetOutputLayout());

        DML_SLICE1_OPERATOR_DESC sliceDesc = {};
        sliceDesc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        sliceDesc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        sliceDesc.DimensionCount = static_cast<uint32_t>(inputWindowOffsets.size());
        sliceDesc.InputWindowOffsets = inputWindowOffsets.data();
        sliceDesc.InputWindowSizes = inputWindowSizes.data();
        sliceDesc.InputWindowStrides = inputWindowStrides.data();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_SLICE1, &sliceDesc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression ReverseSubsequences(
        Expression input,
        Expression sequenceLengths,
        uint32_t axis)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc sequenceLengthsTensor = sequenceLengths.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetOutputLayout());

        DML_REVERSE_SUBSEQUENCES_OPERATOR_DESC reverseDesc = {};
        reverseDesc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        reverseDesc.SequenceLengthsTensor = sequenceLengthsTensor.AsPtr<DML_TENSOR_DESC>();
        reverseDesc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        reverseDesc.Axis = axis;

        detail::NodeOutput* const inputs[] = { input.Impl(), sequenceLengths.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_REVERSE_SUBSEQUENCES, &reverseDesc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression MeanVarianceNormalization(
        Expression input,
        Optional<Expression> scale,
        Optional<Expression> bias,
        bool crossChannel,
        bool normalizeVariance,
        float epsilon,
        DML_OPERATOR_TYPE fusedActivation = DML_OPERATOR_INVALID,
        float fusedActivationParam1 = 0.0f,
        float fusedActivationParam2 = 0.0f)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetOutputLayout());
        TensorDesc scaleTensor;
        TensorDesc biasTensor;

        if (scale)
        {
             scaleTensor = scale->Impl()->GetOutputDesc();
        }
        if (bias)
        {
             biasTensor = bias->Impl()->GetOutputDesc();
        }

        detail::FusedActivationStorage storage;
        dml::FusedActivation fusedActivationData(fusedActivation, fusedActivationParam1, fusedActivationParam2);

        DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.ScaleTensor = scale ? scaleTensor.AsPtr<DML_TENSOR_DESC>() : nullptr;
        desc.BiasTensor = bias ? biasTensor.AsPtr<DML_TENSOR_DESC>() : nullptr;
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.CrossChannel = crossChannel;
        desc.NormalizeVariance = normalizeVariance;
        desc.Epsilon = epsilon;
        desc.FusedActivation = detail::GetFusedActivationPtr(fusedActivationData, &storage);

        detail::NodeOutput* const inputs[] = 
        { 
            input.Impl(), 
            scale ? scale->Impl() : nullptr, 
            bias ? bias->Impl() : nullptr 
        };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression ValueScale2D(
        Expression input,
        float scale,
        Span<const float> bias)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetOutputLayout());

        DML_VALUE_SCALE_2D_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.Scale = scale;
        desc.ChannelCount = static_cast<uint32_t>(bias.size());
        desc.Bias = bias.data();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_VALUE_SCALE_2D, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression AveragePooling(
        Expression input,
        Span<const uint32_t> strides,
        Span<const uint32_t> windowSizes,
        Span<const uint32_t> startPadding,
        Span<const uint32_t> endPadding,
        bool includePadding)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();

        assert(strides.size() == windowSizes.size());
        assert(strides.size() == startPadding.size());
        assert(strides.size() == endPadding.size());

        // Calculate output size
        TensorDesc::Dimensions outputSizes;
        outputSizes.push_back(inputTensor.sizes[0]); // N
        outputSizes.push_back(inputTensor.sizes[1]); // C
        for (size_t i = 0; i < windowSizes.size(); ++i)
        {
            uint32_t paddedInputSize = inputTensor.sizes[2 + i] + startPadding[i] + endPadding[i];
            uint32_t outputSize = (paddedInputSize - windowSizes[i]) / strides[i] + 1;
            outputSizes.push_back(outputSize);
        }

        TensorDesc outputTensor(inputTensor.dataType, outputSizes, builder->GetOutputLayout());

        DML_AVERAGE_POOLING_OPERATOR_DESC averagePoolDesc = {};
        averagePoolDesc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        averagePoolDesc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        averagePoolDesc.DimensionCount = static_cast<uint32_t>(windowSizes.size());
        averagePoolDesc.Strides = strides.data();
        averagePoolDesc.WindowSize = windowSizes.data();
        averagePoolDesc.StartPadding = startPadding.data();
        averagePoolDesc.EndPadding = endPadding.data();
        averagePoolDesc.IncludePadding = includePadding;

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_AVERAGE_POOLING, &averagePoolDesc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    // Reinterprets the memory of a tensor with a different type and dimensions (analogously to using
    // reinterpret_cast to access raw bits). Note that this is different to the DML Cast operator, which performs
    // a type cast on the contents of a tensor (analogously to static_cast). The total tensor size of the output
    // (which depends on the supplied type/sizes/strides) must match the input.
    inline Expression Reinterpret(
        Expression input,
        DML_TENSOR_DATA_TYPE newType,
        TensorDesc::Dimensions newSizes,
        Optional<TensorDesc::Dimensions> newStrides)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc newTensor(newType, std::move(newSizes), std::move(newStrides));

        assert(inputTensor.totalTensorSizeInBytes == newTensor.totalTensorSizeInBytes);

        detail::NodeID node = builder->CreateReinterpretNode(input.Impl());
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(newTensor));

        return output;
    }

    // Same as Reinterpret above, but only adjusts tensor dimensions without affecting type.
    inline Expression Reinterpret(
        Expression input,
        TensorDesc::Dimensions newSizes,
        Optional<TensorDesc::Dimensions> newStrides)
    {
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();

        return Reinterpret(input, inputTensor.dataType, std::move(newSizes), std::move(newStrides));
    }

    // Same as Reinterpret above, but only adjusts tensor type without affecting sizes or strides.
    inline Expression Reinterpret(Expression input, DML_TENSOR_DATA_TYPE newType)
    {
        TensorDesc inputTensor = input.Impl()->GetOutputDesc();

        return Reinterpret(input, newType, inputTensor.sizes, inputTensor.strides);
    }

    inline Expression Slice(Expression input, Span<const uint32_t> offsets, Span<const uint32_t> sizes, Span<const uint32_t> strides)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();

        assert(!offsets.empty());
        assert(!sizes.empty());
        assert(!strides.empty());
        assert(offsets.size() == sizes.size() && sizes.size() == strides.size());

        uint32_t dims = static_cast<uint32_t>(sizes.size());

        TensorDesc::Dimensions outputSizes(sizes.begin(), sizes.end());

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, outputSizes, builder->GetOutputLayout());

        DML_SLICE_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.DimensionCount = dims;
        desc.Offsets = offsets.data();
        desc.Sizes = sizes.data();
        desc.Strides = strides.data();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_SLICE, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    inline Expression Tile(Expression input, Span<const uint32_t> repeats)
    {
        detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
        TensorDesc::Dimensions outputSizes = input.GetOutputDesc().sizes;

        assert(repeats.size() == outputSizes.size());

        for (size_t i = 0; i < repeats.size(); ++i)
        {
            outputSizes[i] *= repeats[i];
        }

        TensorDesc inputTensor = input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, outputSizes, builder->GetOutputLayout());

        DML_TILE_OPERATOR_DESC desc = {};
        desc.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
        desc.RepeatsCount = static_cast<uint32_t>(repeats.size());
        desc.Repeats = repeats.data();

        detail::NodeOutput* const inputs[] = { input.Impl() };
        detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_TILE, &desc, inputs);
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(outputTensor));

        return output;
    }

    // Operator overloads for convenience, which merely map to one of the functions above
    inline Expression operator+(Expression a, Expression b) { return dml::Add(a, b); }
    inline Expression operator-(Expression a, Expression b) { return dml::Subtract(a, b); }
    inline Expression operator*(Expression a, Expression b) { return dml::Multiply(a, b); }
    inline Expression operator/(Expression a, Expression b) { return dml::Divide(a, b); }
    inline Expression operator%(Expression a, Expression b) { return dml::ModulusTruncate(a, b); }
    inline Expression operator&(Expression a, Expression b) { return dml::BitAnd(a, b); }
    inline Expression operator|(Expression a, Expression b) { return dml::BitOr(a, b); }
    inline Expression operator^(Expression a, Expression b) { return dml::BitXor(a, b); }
    inline Expression operator<<(Expression a, Expression b) { return dml::BitShiftLeft(a, b); }
    inline Expression operator>>(Expression a, Expression b) { return dml::BitShiftRight(a, b); }
    inline Expression& operator+=(Expression& a, Expression b) { a = a + b; return a; }
    inline Expression& operator-=(Expression& a, Expression b) { a = a - b; return a; }
    inline Expression& operator*=(Expression& a, Expression b) { a = a * b; return a; }
    inline Expression& operator/=(Expression& a, Expression b) { a = a / b; return a; }
    inline Expression& operator%=(Expression& a, Expression b) { a = a % b; return a; }
    inline Expression& operator&=(Expression& a, Expression b) { a = a & b; return a; }
    inline Expression& operator|=(Expression& a, Expression b) { a = a | b; return a; }
    inline Expression& operator^=(Expression& a, Expression b) { a = a ^ b; return a; }
    inline Expression& operator<<=(Expression& a, Expression b) { a = a << b; return a; }
    inline Expression& operator>>=(Expression& a, Expression b) { a = a >> b; return a; }

    // Operations involving scalars can be reduced to elementwise identity
    inline Expression operator+(Expression a, float b) { return dml::Identity(a, DML_SCALE_BIAS{ 1.0f, b }); }
    inline Expression operator-(Expression a, float b) { return dml::Identity(a, DML_SCALE_BIAS{ 1.0f, -b }); }
    inline Expression operator*(Expression a, float b) { return dml::Identity(a, DML_SCALE_BIAS{ b, 0.0f }); }
    inline Expression operator/(Expression a, float b) { return dml::Identity(a, DML_SCALE_BIAS{ 1.0f / b, 0.0f }); }
    inline Expression operator+(float a, Expression b) { return dml::Identity(b, DML_SCALE_BIAS{ 1.0f, a }); }
    inline Expression operator-(float a, Expression b) { return dml::Identity(b, DML_SCALE_BIAS{ -1.0f, a }); }
    inline Expression operator*(float a, Expression b) { return dml::Identity(b, DML_SCALE_BIAS{ a, 0.0f }); }
    inline Expression operator/(float a, Expression b) { return dml::Recip(b, DML_SCALE_BIAS{ a, 0.0f }); }
    inline Expression& operator+=(Expression& a, float b) { a = a + b; return a; }
    inline Expression& operator-=(Expression& a, float b) { a = a - b; return a; }
    inline Expression& operator*=(Expression& a, float b) { a = a * b; return a; }
    inline Expression& operator/=(Expression& a, float b) { a = a / b; return a; }

    // Unary
    inline Expression operator~(Expression input) { return dml::BitNot(input); }
    inline Expression operator+(Expression input) { return dml::Identity(input); }
    inline Expression operator-(Expression input)
    {
        DML_SCALE_BIAS scaleBias = {};
        scaleBias.Scale = -1.0f;
        return dml::Identity(input, scaleBias);
    }

    // Logical
    inline Expression operator!(Expression a) { return dml::LogicalNot(a); }
    inline Expression operator&&(Expression a, Expression b) { return dml::LogicalAnd(a, b); }
    inline Expression operator||(Expression a, Expression b) { return dml::LogicalOr(a, b); }
    inline Expression operator>(Expression a, Expression b) { return dml::LogicalGreaterThan(a, b); }
    inline Expression operator<(Expression a, Expression b) { return dml::LogicalLessThan(a, b); }
    inline Expression operator==(Expression a, Expression b) { return dml::LogicalEquals(a, b); }
    inline Expression operator!=(Expression a, Expression b) { return !(a == b); }
    inline Expression operator>=(Expression a, Expression b) { return a > b || a == b; }
    inline Expression operator<=(Expression a, Expression b) { return a < b || a == b; }

    // GraphBuilder implementation details
    namespace detail
    {
        inline NodeID GraphBuilder::CreateOperatorNode(
            DML_OPERATOR_TYPE type,
            const void* desc,
            Span<NodeOutput* const> inputs)
        {
            DML_OPERATOR_DESC opDesc = { type, desc };

            Microsoft::WRL::ComPtr<IDMLOperator> op;
            DMLX_THROW_IF_FAILED(m_device->CreateOperator(&opDesc, IID_PPV_ARGS(&op)));

            OperatorNode node = {};
            node.op = std::move(op);
            node.inputs.assign(inputs.begin(), inputs.end());

            uint32_t index = static_cast<uint32_t>(m_operatorNodes.size());
            m_operatorNodes.push_back(std::move(node));

            return { NodeType::Operator, index };
        }

        inline NodeID GraphBuilder::CreateInputNode(uint32_t inputIndex)
        {
            uint32_t index = static_cast<uint32_t>(m_inputNodes.size());
            m_inputNodes.push_back(InputNode{ inputIndex });
            return { NodeType::Input, index };
        }

        inline NodeID GraphBuilder::CreateReinterpretNode(NodeOutput* input)
        {
            uint32_t index = static_cast<uint32_t>(m_reinterpretNodes.size());
            m_reinterpretNodes.push_back(ReinterpretNode{ input });
            return { NodeType::Reinterpret, index };
        }

        inline NodeOutput* GraphBuilder::CreateNodeOutput(NodeID node, uint32_t outputIndex, TensorDesc tensorDesc)
        {
            // Construct the object in the deque, which doesn't invalidate references to elements as it grows
            m_nodeOutputs.emplace_back(this, node, outputIndex, std::move(tensorDesc));

            return &m_nodeOutputs.back();
        }

        inline GraphDesc GraphBuilder::GetGraphDesc(Span<const Expression> outputs) const
        {
            GraphDesc desc = {};
            desc.inputCount = static_cast<uint32_t>(m_inputNodes.size());
            desc.outputCount = static_cast<uint32_t>(outputs.size());

            for (const OperatorNode& node : m_operatorNodes)
            {
                uint32_t nodeIndex = static_cast<uint32_t>(desc.nodes.size());
                desc.nodes.push_back(DML_OPERATOR_GRAPH_NODE_DESC{ node.op.Get() });

                // Walk through each of this node's inputs and add it as an edge
                const uint32_t inputCount = static_cast<uint32_t>(node.inputs.size());
                for (uint32_t inputIndex = 0; inputIndex < inputCount; ++inputIndex)
                {
                    NodeOutput* input = node.inputs[inputIndex];
                    if (input == nullptr)
                    { 
                        continue;
                    }
                    NodeID inputNode = input->GetNode();
                    
                    // Reinterpret nodes aren't "real" nodes, they're just used to modify TensorDescs across
                    // edges. So we follow this node backwards until it hits a real node.
                    while (inputNode.type == NodeType::Reinterpret)
                    {
                        input = m_reinterpretNodes[inputNode.index].input;
                        inputNode = input->GetNode();
                    }

                    if (inputNode.type == NodeType::Input)
                    {
                        DML_INPUT_GRAPH_EDGE_DESC inputEdge = {};
                        inputEdge.GraphInputIndex = m_inputNodes[inputNode.index].inputIndex;
                        inputEdge.ToNodeIndex = nodeIndex;
                        inputEdge.ToNodeInputIndex = inputIndex;

                        desc.inputEdges.push_back(inputEdge);
                    }
                    else if (inputNode.type == NodeType::Operator)
                    {
                        DML_INTERMEDIATE_GRAPH_EDGE_DESC intermediateEdge = {};
                        intermediateEdge.FromNodeIndex = inputNode.index;
                        intermediateEdge.FromNodeOutputIndex = input->GetOutputIndex();
                        intermediateEdge.ToNodeIndex = nodeIndex;
                        intermediateEdge.ToNodeInputIndex = inputIndex;

                        desc.intermediateEdges.push_back(intermediateEdge);
                    }
                    else
                    {
                        assert(false); // Invalid node type
                        DMLX_THROW(E_UNEXPECTED);
                    }
                }
            }

            // Add output edges
            for (uint32_t outputIndex = 0; outputIndex < desc.outputCount; ++outputIndex)
            {
                NodeOutput* output = outputs[outputIndex].Impl();
                if (output == nullptr)
                {
                    continue;
                }
                NodeID outputNode = output->GetNode();
                
                // Reinterpret nodes are meaningless on outputs (they're no-ops), so just follow them back until we
                // get to a real operator node.
                while (outputNode.type == NodeType::Reinterpret)
                {
                    output = m_reinterpretNodes[outputNode.index].input;
                    outputNode = output->GetNode();
                }

                if (outputNode.type == NodeType::Input)
                {
                    // It's not valid to connect an output of the graph directly to an input without an intervening
                    // node. If this behavior is desired, it should instead be accomplished with a copy e.g. using
                    // the elementwise identity operator.
                    DMLX_THROW(E_INVALIDARG);
                }

                assert(outputNode.type == NodeType::Operator);

                DML_OUTPUT_GRAPH_EDGE_DESC outputEdge = {};
                outputEdge.FromNodeIndex = output->GetNode().index;
                outputEdge.FromNodeOutputIndex = output->GetOutputIndex();
                outputEdge.GraphOutputIndex = outputIndex;

                desc.outputEdges.push_back(outputEdge);
            }

            // Sanity
            assert(desc.nodes.size() == m_operatorNodes.size());
            assert(desc.outputEdges.size() == desc.outputCount);
            assert(desc.outputCount == outputs.size());

            return desc;
        }
    } // namespace detail

} // namespace dml
