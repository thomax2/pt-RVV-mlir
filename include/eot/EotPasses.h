#ifndef EOT_EOTPASSES_H
#define EOT_EOTPASSES_H
#include "mlir/Pass/Pass.h"
#include <memory>
namespace eot {
#define GEN_PASS_DECL
#include "eot/EotPasses.h.inc"
std::unique_ptr<mlir::Pass> createConvertTosaCustomToEotPass();
std::unique_ptr<mlir::Pass> createConvertEotToStandardPass();
std::unique_ptr<mlir::Pass> createPlanStaticWorkspacePass();
#define GEN_PASS_REGISTRATION
#include "eot/EotPasses.h.inc"
}
#endif
